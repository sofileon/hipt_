import cv2
import numpy as np


class Contour_Checking_fn(object):
    # Defining __call__ method
    def __call__(self, pt):
        raise NotImplementedError


class isInContourV1(Contour_Checking_fn):
    def __init__(self, contour):
        self.cont = contour

    def __call__(self, pt):
        return (
            1
            if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False)
            >= 0
            else 0
        )


class isInContourV2(Contour_Checking_fn):
    def __init__(self, contour, patch_size):
        self.cont = contour
        self.patch_size = patch_size

    def __call__(self, pt):
        pt = np.array(
            (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        ).astype(float)
        return (
            1
            if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False)
            >= 0
            else 0
        )


# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift),
            ]
        else:
            all_points = [center]

        for points in all_points:
            if (
                cv2.pointPolygonTest(
                    self.cont, tuple(np.array(points).astype(float)), False
                )
                >= 0
            ):
                return 1
        return 0


# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class isInContourV3_Hard(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift),
            ]
        else:
            all_points = [center]

        for points in all_points:
            if (
                cv2.pointPolygonTest(
                    self.cont, tuple(np.array(points).astype(float)), False
                )
                < 0
            ):
                return 0
        return 1


class isInContour_pct(Contour_Checking_fn):
    def __init__(
        self, contour, contour_holes, tissue_mask, patch_size, scale, pct=0.01
    ):
        self.cont = contour
        self.holes = contour_holes
        self.mask = tissue_mask // 255
        self.patch_size = patch_size
        self.scale = scale
        self.pct = pct

    def __call__(self, pt):

        # work on downsampled image to compute tissue percentage
        downsampled_patch_size = int(self.patch_size * 1 / self.scale[0])
        downsampled_pt = pt * 1 / self.scale[0]
        x_patch, y_patch = downsampled_pt
        x_patch, y_patch = int(x_patch), int(y_patch)

        # draw white filled contour on black background
        contour_mask = np.zeros_like(self.mask)
        cv2.drawContours(contour_mask, [self.cont], 0, (255, 255, 255), -1)

        # draw black filled holes on white filled contour
        cv2.drawContours(contour_mask, self.holes, 0, (0, 0, 0), -1)

        # apply mask to input image
        mask = cv2.bitwise_and(self.mask, contour_mask)

        # x,y axis inversed
        sub_mask = mask[
            y_patch : y_patch + downsampled_patch_size,
            x_patch : x_patch + downsampled_patch_size,
        ]

        patch_area = downsampled_patch_size**2
        tissue_area = np.sum(sub_mask)
        tissue_pct = tissue_area / patch_area

        if tissue_pct >= self.pct:
            return 1
        else:
            return 0
