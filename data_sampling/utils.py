import os
import sys
import torch
import tqdm
import h5py
import patchify
import random
import csv
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from typing import Callable, Dict, Optional
from torch.utils.data import DataLoader
from torchvision import datasets
from source.utils import update_state_dict


def load_model(model, checkpoint_path, loc, checkpoint_key='teacher'):
    state_dict = torch.load(checkpoint_path, map_location=loc)
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict, msg = update_state_dict(model.state_dict(), state_dict)
        model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights found at {checkpoint_path}")
        print(msg)
    return model

def patch_feature_extraction(model,dataset,device,save_features=True,output_features_dir=None):
    label2num_mapping = dataset.class_to_idx 
    num2label_mapping = {v: k for k, v in label2num_mapping.items()} 
    if save_features:
        if output_features_dir is None:
            raise ValueError("output_features_dir must be provided if save_features is True")

    params_instance = {'batch_size': 1,
                'shuffle': False,
                'pin_memory': False,
                'num_workers': 4}  
    dataLoader = DataLoader(dataset, **params_instance)
    model = model.to(device)
    model.eval()
    feature_matrix = []
    labels = []
    with torch.no_grad():
        with tqdm.tqdm(
            dataLoader,
            total=len(dataLoader),
            desc=(f"Patch-level feature extraction"),
            unit=" patches",
            unit_scale=dataLoader.batch_size,
            ncols=100,
            leave=True,
            file=sys.stdout,
        ) as t:
            for i, (patch_, label_num) in enumerate(t):
                patch_, label_num = patch_.to(device, non_blocking=True), label_num.to(device, non_blocking=True)
                q = model(patch_)
                q = q.squeeze().cpu().numpy()
                label=num2label_mapping.get(int(label_num.cpu()))
                labels = np.append(labels, label)
                if save_features:
                    slide_features_dir=Path(f'{output_features_dir}/{label}') #if the dataset is labeled, save the features in a folder with the label name
                    slide_features_dir.mkdir(parents=True,exist_ok=True)
                    save_path = Path(slide_features_dir, f"{Path(dataset.imgs[i][0]).stem}.pt") #save the feature file with the same name as the patch file
                    torch.save(q, save_path)
                feature_matrix = np.append(feature_matrix, q)
    feature_matrix = feature_matrix.reshape(len(dataset), model.embed_dim)
    return feature_matrix, labels
    
def load_features(features_dir):
    dataset = datasets.DatasetFolder(features_dir, loader=lambda x: torch.load(x,map_location='cpu'), extensions=('.pt'))
    label2num_mapping = dataset.class_to_idx
    num2label_mapping = {v: k for k, v in label2num_mapping.items()}
    dataLoader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    feature_matrix = []
    labels = []
    with torch.no_grad():
        with tqdm.tqdm(
            dataLoader,
            total=len(dataLoader),
            desc=(f"Patch-level feature loading"),
            unit=" patches",
            unit_scale=dataLoader.batch_size,
            ncols=100,
            leave=True,
            file=sys.stdout,
        ) as t:
            for feature, label_num in t:
                feature = feature.numpy()
                embed_dim = feature.shape[1]
                label=[num2label_mapping.get(int(l)) for l in label_num.cpu()]
                labels = np.append(labels, label)
                try:
                    feature_matrix = np.append(feature_matrix, feature, axis=0)
                except ValueError:
                    feature_matrix = feature
    feature_matrix = feature_matrix.reshape(len(dataset), embed_dim)
    return feature_matrix, labels

def ClusterIndicesNumpy(clustNum, labels_array):
    #get the indexes of where labels are a given number clustNum
    return np.where(labels_array == clustNum)[0]

def LabelsInCluster(clustNum, kmeans_labels, target_list, verbose=False):
    """Returns the label with most occurences in a cluster

    Args:
        clustNum (int): Cluster number encoding
        kmeans_labels (lst): k-means label prediction  
        target_list (lst): list with groundtrith labels for the fitted kmeans
        verbose (bool, optional): Wheter or not to display the percentages of each class in cluster clustNum. Defaults to False.

    Returns:
        str: the groundtruth label with most appearances for cluster clustNum 
    """    
    target_list=np.array(target_list)[ClusterIndicesNumpy(clustNum, kmeans_labels)] #Get the groundtruth labels of the features from cluster clustNum
    t_dict ={l:np.count_nonzero(target_list==l) for l in set(target_list)} #dictionary with the groundtruth labels as key and their num of occurences as value
    print(f'Image Class with more occurences ({max(t_dict.values())}/{target_list.shape[0]}) in the cluster {clustNum}:{max(t_dict, key=t_dict.get)} ({max(t_dict.values())*100/target_list.shape[0]:.2f}% of the cluster)')
    if verbose:
        for k,v in t_dict.items():
            print(f'{k}: {v} ({v*100/target_list.shape[0]:.2f}%)')
    return max(t_dict, key=t_dict.get)

def SamplingFromClusters(clustInd, y=None, sampling_percentage=1, n_sample=None, verbose=False):
    if n_sample is None:
        n_sample = int(sampling_percentage * len(clustInd))
    if verbose:
        print(f'Extracting {n_sample} samples from this cluster')
    idxs = random.sample(range(len(clustInd)), k=n_sample)
    if y is not None:
        y_cluster=np.array(y)[clustInd]
        return clustInd[idxs],y_cluster[idxs]
    else:
        return clustInd[idxs]
    
class HierarchicalPretrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features_dir: str,
        labeled_dataset=False,
        transform: Callable= None,
    ):
        self.features_list = [f for f in Path(features_dir).glob("**/*.pt")]
        self.features_list.sort()
        self.labeled_dataset=labeled_dataset
        if self.labeled_dataset:
        #get the label of the features
            self.classes = sorted(entry.name for entry in os.scandir(features_dir) if entry.is_dir())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.transform = transform

    def __getitem__(self, idx: int):
        f = torch.load(self.features_list[idx],map_location='cpu')
        if self.transform is not None:
            f = self.transform(f)
        if self.labeled_dataset:
            label = self.class_to_idx[Path(self.features_list[idx]).parent.name]
        else:
            label =torch.zeros(1, 1)
        return f, label

    def __len__(self):
        return len(self.features_list)

class SamplingDataAugmentation(object):
    def __init__(
            self,
            device
        ):
        self.device=device

    def __call__(self, x):
        x = x.detach().to(self.device)
        return x

# find the h5 file with the same name as the feature file
def get_region_from_h5file(feature_path, h5_files_list):
    name = Path(feature_path).stem
    corresponding_img=[h5_file for h5_file in h5_files_list if name in Path(h5_file).stem][0]
    #open h5 file
    # region_size = h5py.File(corresponding_img, 'r')['imgs'].attrs['patch_size']
    # patch_size = h5py.File(corresponding_img, 'r')['imgs'].attrs['subpatch_size']
    # img=h5py.File(corresponding_img, 'r')['imgs'][:].reshape(region_size//patch_size, region_size//patch_size, 1, patch_size, patch_size, 3)
    # img = Image.fromarray(patchify.unpatchify(img, (region_size, region_size, 3)))
    return corresponding_img

def get_patch_from_h5file(args):
    experiment_name,sampling_cluster, feature_path, h5_files_list, patch_index, output_dir = args
    name = Path(feature_path).stem # name of the feature file 
    corresponding_img=[h5_file for h5_file in h5_files_list if name in Path(h5_file).stem][0]
    #open h5 file
    img=Image.fromarray(h5py.File(corresponding_img, 'r')['imgs'][patch_index])
    output_dir.mkdir(exist_ok=True, parents=True)
    img.save(output_dir / f"{name}_{patch_index}.png")
    csv_writer(Path(output_dir).parent, experiment_name, sampling_cluster, corresponding_img,patch_index)
    return name

def csv_writer(output_dir,experiment_name, cluster,region_path, patch_index):
    #check if csv exists
    csv_path=Path(f'{output_dir}/{experiment_name}_sampled_patches.csv')
    if not csv_path.is_file():
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Cluster','Region_path','Patch_index'])
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([cluster, region_path, patch_index])