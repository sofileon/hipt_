import h5py
import argparse
import numpy as np
import multiprocessing as mp
from PIL import Image
from pathlib import Path, PurePath
import csv
import time
import os
this_path=Path(__file__).resolve()

#Read all the hdf5 file in a folder and save them as jpegs
def image_extraction(args):
    output_dir, h5_file = args
    output_dir.mkdir(parents=True, exist_ok=True)
    images=h5py.File(h5_file,'r')['imgs'][:]# [256x256x256x3] np array
    coord = h5py.File(h5_file,'r')['coords'][:] # [1x2] np array
    patch_coords=[(x,y) for y in range(coord[1],coord[1]+4096,256) for x in range(coord[0],coord[0]+4096,256)]
    for img,coords in zip(images,patch_coords):
        name=f'{str(output_dir.stem)}_{coords[0]}_{coords[1]}'
        img=Image.fromarray(img)
        img.save(Path(output_dir,f'{name}.jpeg'))
    return output_dir

        
def main(args):
    head_parts=PurePath(this_path).parts[:3]
    running_in=Path(*head_parts)
    h5_files_dir= Path(running_in,'projects/pathology-self-supervision/data/examode_colon/region_4096_h5/',args.h5_files_dir)
    processed_list =  Path(running_in,'projects/pathology-self-supervision/data/examode_colon/processed_region2patch_jpeg.csv')
    if not processed_list.is_file():
        with open(processed_list, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['WSI','Label','Num_patches'])
    list_colon_folders = [f.path for f in os.scandir(h5_files_dir) if f.is_dir()]
    list_colon_folders.sort()
    print(f'Processing {len(list_colon_folders)} WSI.')
    class_time=0
    for index,colon_folder in enumerate(list_colon_folders[args.start_point:], start=args.start_point):
        with open(processed_list, 'r') as f:
            reader = csv.reader(f)
            processed = [row[0] for row in reader]
            processed.remove('WSI')
        if Path(colon_folder).stem in processed:
            print(f'({index}): {Path(colon_folder).stem} already processed.')
            continue
        start_time = time.time()
        output_dir = Path(Path(colon_folder).parent.parent.parent,f'region_4096_jpeg/{Path(colon_folder).parent.stem}')
        h5_files=[Path(wsi_dir[0],region) for wsi_dir in os.walk(colon_folder,followlinks=True) for region in wsi_dir[2]]
        num_workers = mp.cpu_count()
        if num_workers > 4: 
            num_workers = 4
        with mp.Pool(num_workers) as pool:
            iterable = [(Path(output_dir,h5_f.parent.stem), h5_f) for h5_f in h5_files]
            results = pool.map(image_extraction, iterable)
        total_time = time.time() - start_time
        class_time += total_time
        print(f'({index}): Finished processing {len(h5_files)} regions from {Path(colon_folder).stem} in {time.strftime("%H:%M:%S", time.gmtime(total_time))}.')
        with open(processed_list, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([Path(colon_folder).stem,args.h5_files_dir,len(results)*256])
    print(f'Finished processing {len(list_colon_folders)} ({args.h5_files_dir}) WSI in {time.strftime("%H:%M:%S", time.gmtime(class_time))} (hh:mm:ss).')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_files_dir', type=str,help='colon directory', default='colon_cancer_hgd')
    parser.add_argument('--start_point', type=int,help='start point', default=0)
    args = parser.parse_args()
    #   python3 HDF5_creation/image_extraction.py --h5_files_dir=colon_cancer_hgd
    main(args)
