import glob
import csv
import os
import torch
import hydra
import time
import shutil
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from source.vision_transformer import vit_small
import tqdm
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from utils import *


def h52im(experiment_name, cluster, file_indexes, patch_indexes, feature_paths, h5_file_paths, out_path):
        num_workers = mp.cpu_count()
        if num_workers > 4: 
            num_workers = 4
        with mp.Pool(num_workers) as pool:
            iterable = [(experiment_name, cluster,feature_paths[file], h5_file_paths, patch, out_path) for file,patch in zip(file_indexes,patch_indexes)]
            results = pool.map(get_patch_from_h5file, iterable)
        return results


@hydra.main(
    version_base="1.2.0", config_path="../config/datasampling", config_name="CRC_kmeans"
)
def main(cfg:DictConfig):
    log_path=Path(f'{cfg.output_dir}/{cfg.experiment_name}_log.csv')
    if not cfg.resume:
        if log_path.exists():
            os.remove(log_path)
        if Path(cfg.output_dir, cfg.experiment_name).exists():
            shutil.rmtree(Path(cfg.output_dir, cfg.experiment_name))
        if cfg.feature_extraction:
            print(f'Extracting features:{cfg.feature_extraction}')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            dataset = datasets.ImageFolder(cfg.patch_files_dir, transform=transform)
            vit_patch = vit_small(
                img_size=cfg.model_params.patch_size,
                patch_size=cfg.model_params.mini_patch_size,
                embed_dim=cfg.model_params.embed_dim_patch
            )
            device256=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            vit_patch = load_model(vit_patch, cfg.model_params.weights256, device256)
            feature_matrix, labels = patch_feature_extraction(vit_patch, dataset, device256, cfg.save_features, cfg.output_features_dir)
        
        if cfg.method=='kmeans':
            print('Doing kmeans')
            if not cfg.feature_extraction:
                feature_matrix, labels = load_features(cfg.patch_features_dir) 
            t0 = time.time()
            kmeans = KMeans(**cfg.kmeans).fit(feature_matrix)
            fit_time = time.time() - t0
            print(f'Time to fit kmeans:{fit_time:.3f}s')
            kmeans_centers=kmeans.cluster_centers_
            if cfg.labeled_features:
                labels=np.array(labels)
                labels_kmeans=[]
                for i in range(cfg.kmeans.n_clusters):
                    label_kmean=LabelsInCluster(i,kmeans.labels_, labels)
                    # logging
                    with open(log_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([f'Cluster_{i} ',label_kmean])
                    labels_kmeans.append(label_kmean)
                print(labels_kmeans)          
                dict_counts={}
                for label in np.unique(labels):
                    dict_counts[label]=[]
                for i in range(cfg.kmeans.n_clusters):
                    x_clus=SamplingFromClusters(ClusterIndicesNumpy(i,kmeans.labels_),y=labels, sampling_percentage=1)
                    for label in np.unique(labels):
                        dict_counts[label].append(np.count_nonzero(np.array(x_clus[1])==label))
                count_df=pd.DataFrame.from_dict(dict_counts)
                count_df.to_csv(f'{cfg.output_dir}/{cfg.experiment_name}_kmeans_cluster_count.csv')
        
        elif cfg.method == 'mini_batch_kmeans':
            print('Doing mini batch kmeans')
            if cfg.feature_extraction:
                # the directory where the extracted features where just saved, provided in the config file
                features_dir=cfg.output_features_dir 
            elif not cfg.feature_extraction:
                features_dir=cfg.patch_features_dir
            dataset = HierarchicalPretrainingDataset(features_dir,labeled_dataset=cfg.labeled_features, transform=None)
            if cfg.labeled_features:
                label2num_mapping = dataset.class_to_idx 
            print(f'Number of feature files for mini-batch kmeans:{len(dataset)}')
            data_loader = torch.utils.data.DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=cfg.mini_batch_kmeans.batch_size//cfg.features_per_file,
                    num_workers=4
            )
            kmeans= MiniBatchKMeans(**cfg.mini_batch_kmeans)
            t0 = time.time()
            with tqdm.tqdm(
                    data_loader,
                    desc=(f"Fitting mini batch kmeans:"),
                    unit=" batch",
                ) as t:
                for i, (region_features, _) in enumerate(t):
                    if region_features.dim()>2:
                        region_features=torch.flatten(region_features,start_dim=0, end_dim=1) #(batch_size,256, n_features)->(batch_size*256, n_features)
                    kmeans = kmeans.partial_fit(region_features)
            fit_time = time.time() - t0
            print(f'Time to fit mini-batch kmeans:{fit_time:.3f}s')     
            kmeans_centers=kmeans.cluster_centers_
            if cfg.labeled_features:
                num2label_mapping = {v: k for k, v in label2num_mapping.items()} 
                kmeans_pred=[]
                labels=[]
                # get predictions for all the features used for fit kmeans
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=cfg.mini_batch_kmeans.batch_size//cfg.features_per_file,
                    num_workers=4
                )
                for i, (region_features,label_num) in enumerate(data_loader):
                    if region_features.dim()>2:
                        region_features=torch.flatten(region_features,start_dim=0, end_dim=1) 
                    kmeans_pred.extend(kmeans.predict(region_features))
                    label= [num2label_mapping.get(int(l)) for l in label_num.cpu()]
                    labels = np.append(labels, label)
                labels=np.array(labels)
                kmeans_pred = np.array(kmeans_pred)
                labels_kmeans=[]
                for i in range(cfg.kmeans.n_clusters):
                    label_kmean=LabelsInCluster(i,kmeans_pred, labels)
                    # logging
                    with open(log_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([f'Cluster_{i} ',label_kmean])
                    labels_kmeans.append(label_kmean)
                print(labels_kmeans)
                dict_counts={}
                for label in np.unique(labels):
                    dict_counts[label]=[]
                for i in range(cfg.kmeans.n_clusters):
                    x_clus=SamplingFromClusters(ClusterIndicesNumpy(i,kmeans_pred),y=labels, sampling_percentage=1)
                    for label in np.unique(labels):
                        dict_counts[label].append(np.count_nonzero(np.array(x_clus[1])==label))
                count_df=pd.DataFrame.from_dict(dict_counts)
                count_df.to_csv(f'{cfg.output_dir}/{cfg.experiment_name}_kmeans_cluster_count.csv')
        # logging
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f'Time to fit {cfg.method}',fit_time])
        
        # load the dataset that will be sampled
        transform=SamplingDataAugmentation('cpu')
        sampling_dataset = HierarchicalPretrainingDataset(cfg.feature_files_dir, labeled_dataset=False, transform=transform)
        h5_paths = glob.glob(os.path.join(cfg.h5_files_dir, '**/*.h5'), recursive=True)
        h5_paths.sort()
        print(f'Number of h5 files:{len(h5_paths)}')
        dataLoader_sampling = DataLoader(sampling_dataset, batch_size=4, shuffle=False, num_workers=4)
        all_pred=[]
        t0 = time.time()
        with tqdm.tqdm(
                dataLoader_sampling,
                desc=(f"Predicting kmeans:"),
                unit=" features",
                unit_scale=dataLoader_sampling.batch_size,
            ) as t:
            for i, (region_features, _) in enumerate(t):
                region_features=torch.flatten(region_features,start_dim=0, end_dim=1) #flattens the batch (batch_size,256, n_features) -> (batch_size*256, n_features)
                try:
                    predictions = kmeans.predict(region_features.detach().numpy())
                except ValueError:
                    predictions = kmeans.predict(region_features.detach().numpy().astype('float'))
                all_pred.extend(predictions)
        all_pred=np.array(all_pred)
        pred_time = time.time() - t0
        print(f'Time to predict on {len(all_pred)} patches:{pred_time:.3f}s')
        #get the counts of each cluster on all_pred
        unique, counts = np.unique(all_pred, return_counts=True)
        cluster_counts=dict(zip(unique, counts))
        print(f'Cluster counts:\n{cluster_counts.items()}')
        # logging
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f'Time to predict {cfg.method}',pred_time])
            for k,v in cluster_counts.items():
                writer.writerow([f'{k} count',v])

        for c in range(cfg.kmeans.n_clusters):
            output_dir= Path(cfg.output_dir,cfg.experiment_name)
            sampling_cluster=f'cluster_{c}'
            output_dir=Path(output_dir,f'cluster_{c}')
            output_dir.mkdir(parents=True, exist_ok=True)
            cluster=ClusterIndicesNumpy(c,all_pred)
            #choose the smallest number between the number of samples and the number of patches in the cluster
            n_samples=min(cfg.n_samples,len(cluster))
            print(f'Sampling {n_samples} patches from {sampling_cluster} cluster')
            indexes_sample=SamplingFromClusters(cluster,n_sample=n_samples)
            np.savetxt(f'{output_dir}/feature_indexes.csv', indexes_sample, delimiter=',', fmt='%s')
    if cfg.resume:
        print('Resuming sampling')
        transform=SamplingDataAugmentation('cpu')
        sampling_dataset = HierarchicalPretrainingDataset(cfg.feature_files_dir, labeled_dataset=False, transform=transform)
        h5_paths = glob.glob(os.path.join(cfg.h5_files_dir, '**/*.h5'), recursive=True)
        h5_paths.sort()
    #read log csv file
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        log = list(reader)[11:11+cfg.kmeans.n_clusters]
    #delete count from str
    cluster_names = [cluster[0].split(' count')[0] for cluster in log]
    if len(cluster_names)!=cfg.kmeans.n_clusters:
        raise ValueError(f'Number of clusters in log file:{len(cluster_names)} is different from the number of clusters in the config file:{cfg.kmeans.n_clusters}')
    t0 = time.time()
    print(cluster_names)
    for c in cluster_names:    
        if int(c)<5:
            continue
        output_dir= Path(cfg.output_dir,cfg.experiment_name)
        sampling_cluster=f'cluster_{c}'
        output_dir=Path(output_dir,sampling_cluster)
        #count the number of images in the output_dir
        n_images=len(glob.glob(os.path.join(output_dir, '**/*.png'), recursive=True))
        #read the indexes_sample from the csv
        total_index_samples=np.loadtxt(f'{output_dir}/feature_indexes.csv', delimiter=',', dtype=int)
        indexes_sample=total_index_samples[n_images:]
        print(f'Sampling {len(indexes_sample)}/{len(total_index_samples)} patches from {sampling_cluster}')
        #div by 256 is the index of the .pt path in the dataset and the modulus of 256 is the feature position
        feature_pt_ind=indexes_sample//256
        patch_ind=indexes_sample%256
        results=h52im(cfg.experiment_name, sampling_cluster, feature_pt_ind,patch_ind, sampling_dataset.features_list,h5_paths, output_dir)
        print(f'Extracted {len(results)} images from {sampling_cluster}')
    sampling_time = time.time() - t0
    print(f'Time to sample:{sampling_time:.3f}s')
    with open(log_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['Time to extract patches',sampling_time])

if __name__ == "__main__":
    #   python3 data_sampling/data_sampling.py --config-name 'CRC_kmeans'
    main()