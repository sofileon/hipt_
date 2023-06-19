from pathlib import Path
import random
import sys
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import umap.plot
from source.vision_transformer import vit_small
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets
from source.utils import update_state_dict, fix_random_seeds

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)


thispath = Path(__file__).resolve()


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


@hydra.main(
    version_base="1.2.0", config_path="./config/evaluation/", config_name="CRC_umap"
)
def main(cfg:DictConfig):
    fix_random_seeds(0)
# Load patch vit
    weights256 = cfg.pretrain_vit_patch
    vit_patch = vit_small(
                    img_size=cfg.patch_size, #256
                    patch_size=cfg.mini_patch_size, #16 
                    embed_dim=cfg.embed_dim_patch #384
                )
    device256=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vit_patch = load_model(vit_patch, weights256, device256)
    vit_patch.to(device256)

    print(f"Loaded pretrained ViT256-16  ({cfg.experiment_name})")

    # Load patches
    print(f"Loading data...")
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.stddev)])
    dataset = datasets.ImageFolder(cfg.dataset.data_dir, transform=transform)
    label_mapping = dataset.class_to_idx
    label_mapping = {v: k for k, v in label_mapping.items()}
    if cfg.training.pct:
        print(f"Getting UMAP of {cfg.training.pct*100}% of the data")
        nsample = int(cfg.training.pct * len(dataset))
        idxs = random.sample(range(len(dataset)), k=nsample)
        dataset = torch.utils.data.Subset(dataset, idxs)
        selected_patches_path = [dataset.dataset.imgs[i][0] for i in idxs]

    else:
        selected_patches_path = [dataset.imgs[i][0] for i in range(len(dataset))]

    params_instance = {'batch_size': 1,
                    'shuffle': False,
                    'pin_memory': False,
                    'num_workers': 4}
    
    dataloader = DataLoader(dataset, **params_instance)

    vit_patch.eval()
    feature_dict = {}
    feature_matrix = []
    labels = []
    with torch.no_grad():
        #add tqdm
        with tqdm.tqdm(
            dataloader,
            total=len(dataloader),
            desc=(f"Patch-level feature extraction"),
            unit=" patches",
            unit_scale=dataloader.batch_size,
            ncols=100,
            leave=True,
            file=sys.stdout,
        ) as t:

            for i, (patch_, label_id) in enumerate(t):
                patch_, label_id = patch_.to(device256, non_blocking=True), label_id.to(device256, non_blocking=True)
                feature = vit_patch(patch_)
                feature = feature.squeeze().cpu().numpy()
                label=label_mapping.get(int(label_id.cpu()))
                labels = np.append(labels, label)
                feature_dict[Path(str(selected_patches_path[i])).stem] = feature
                feature_matrix = np.append(feature_matrix, feature)

    # uMap
    feature_matrix = feature_matrix.reshape(len(dataset), cfg.embed_dim_patch)
    encoder_name=  '$ViT_{256}-16$'
    plot_title = f"{cfg.experiment_name} {encoder_name} embeddings"
    image_name = f"{cfg.experiment_name}_uMap_similarity.svg"
    if cfg.pca.enable:
        print(f"Performing PCA with {cfg.pca.n_components} components")
        pca = PCA(n_components=cfg.pca.n_components)
        feature_matrix = pca.fit_transform(feature_matrix)
        print(f"Explained variation per principal component: {pca.explained_variance_ratio_}")
        plot_title = f"{cfg.experiment_name} {encoder_name} embeddings (with PCA)"
        image_name = f"{cfg.experiment_name}_PCA_uMap_similarity.svg"
    plt.figure()
    mapper = umap.UMAP().fit(feature_matrix)
    im = umap.plot.points(mapper, labels=labels, theme='red')
    plt.title(plot_title)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / image_name)

if __name__ == '__main__':
    main()