<h1 align="center">Hierarchical Image Pyramid Transformer</h1>


Re-implementation of original [HIPT](https://github.com/mahmoodlab/HIPT) code.

<p>
   <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
   <a href="https://github.com/PyCQA/pylint"><img alt="empty" src=https://img.shields.io/github/stars/clemsgrs/hs2p?style=social></a>
</p>

## Requirements

- python 3.9+
- install requirements via `pip3 install -r requirements.txt`
- install module via `pip3 install -e .`

## Prerequisite

You need to have extracted square regions from each WSI you intend to train on.<br>
To do so, you can take a look at [HS2P](https://github.com/clemsgrs/hs2p), which segments tissue and extract relevant patches at a given pixel spacing.


Download HIPT pre-trained weights via the following commands:

<details>
<summary>
Download commands
</summary>

```
mkdir checkpoints
cd checkpoints
gdown 1Qm-_XrTMYhu9Hl-4FClaOMuroyWlOAxw
gdown 1A2eHTT0dedHgdCvy6t3d9HwluF8p5yjz
```
</details>

## Using HDF5 files

If you want to use HDF5 files,run HS2P to extract the coordinates of the 4096x4096 regions without saving the images to disk (setting in the config file `save_patches_to_disk` to False). HS2P outputs one .h5 file per slide containing the coordinates of every region found with the slide under the `/patches/` folder. After finishing to run HS2P in all your dataset, follow this steps:

1. Compile the paths to the .h5 files in a csv.

```
file_path=Path(<HS2P_outputdir>, '/patches/<region_size>/<format>/').rglob('*.h5')
h5_files=[x for x in file_path if x.is_file()]
with open(Path(any/path/region_coordinates.csv), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['h5_files'])
    for h5_file in h5_files:
        writer.writerow([h5_file])
```
2. Create a CSV file containing paths to the slides used in HS2P:

```
slide_path
path/to/slide_1.tif
path/to/slide_2.tif
...
```

3. Create a configuration file under `config/hdf5/` (see examples contained in this repository):
```
hdf5_files_csv:'/path/to/the/csv/with/all_h5_paths/region_coordinates.csv'
wsi_files:'/path/to/the/csv/with/wsi_paths.csv' 
output_dir:'any/path/'
processed_wsi_csv:'any/path/processed_wsi2region_h5.csv'
tissue: <tissue_type_you_are_working_with>
region_2_patches:
  patch_size: 256 # patch size for ViT Patch in HIPT
```
4. Run the following command to kick off region extraction into HDF5 files:

```
python3 HDF5_creation/hdf5_creation.py --config-name <config_file>
```

<details>
<summary>
HDF5 region extraction output
</summary>

```
output_dir/
├── tissue/
│     ├── slide_id_0/
│     │     ├── slide_id_0_x0_y0.h5
│     │     ├── slide_id_0_x1_y0.h5
│     │     └── ...
│     ├── slide_id_1/
│     │     ├── slide_id_1_x0_y0.h5
│     │     ├── slide_id_1_x1_y0.h5
│     │     └── ...
│     └── ...
```
</details>

Each .h5 file will contain one region unrolled into 256 patches of size [256x256] and its coordinates.

## Feature Extraction

**1. [Optional] Configure wandb**

If you want to benefit from wandb logging, you need to follow these simple steps:
 - grab your wandb API key under your profile and export
 - run the following command in your terminal: `export WANDB_API_KEY=<your_personal_key>`
 - update wandb parameters in the relevant config files (mainly `project` and `username`)

**2. Extract features**

Your folder containing the extracted square regions should be structured as follow:

<details>
<summary>
Folder structure
</summary>

```bash
region_dir/
├── slide_1/
│     ├── slide_1.h5
│     └── imgs/
│          ├── region_1.fmt
│          ├── region_2.fmt
│          └── ...
├── slide_2/
├── slide_3/
└── ...
```
</details>

Create a configuration file under `config/feature_extraction/` taking inspiration from existing files.<br>
A good starting point is to use the default configuration file `config/default.yaml` where parameters are documented.

- extract region-level features : make sure to set `level: 'global'` in your config and have 2 gpus available.<br>
- extract patch-level features : make sure to set `level: 'local'` in your config and have 1 gpu available..<br>

Then run the following command to kick off feature extraction:

```bash
python3 extract_features.py --config-name <feature_extraction_config_filename>
```

This will produce one .pt file per slide and save it under `output/<experiment_name>/<level>/`:

```
hipt/
├── output/<experiment_name>/<experiment_id>/features
│     ├── level/
│     │    ├── slide_1.pt
│     │    ├── slide_2.pt
│     │    └── ...
│     └── process_list_<level>.csv
```

## HIPT Training

**1. Prepare `train.csv` and `tune.csv`**

For this pipeline you will need two csv files: `train.csv` and `tune.csv`.<br>
The syntax is easy:

```
slide_id,label
TRAIN_1,1
TRAIN_2,0
...
```

If you want to run testing at the end, you can provide a `test.csv` file.<br>

**2. Train a *single-fold* model on extracted features**

Once features have been extracted, create a configuration file under:

- `config/training/subtyping` for training a subtyping model
- `config/training/survival` for training a survival model

You can take inspiration from `single.yaml` files.<br>
Dump in there the paths to your `train.csv` and `tune.csv` files.<br>
If you want to run testing as well, add the path to your `test.csv` file. Otherwise, leave it blank (it'll skip testing).

If you train the top Transformer block only (i.e. leveraging global features), you only need 1 gpu.
If you train the top & the intermediate Transformer blocks (i.e. leveraging local features), you'll need 2 gpus.

Then, run the following command to kick off model training on a single fold:

- subtyping: `python3 train/subtyping.py --config-name <subtyping_single_fold_config_filename>`
- survival: `python3 train/survival.py --config-name <survival_single_fold_config_filename>`

**3. Train a *multi-fold* model on extracted features**

Your multiple folds should be structured as follow:

<details>
<summary>
Folds structure
</summary>

```bash
fold_dir/
├── fold_1/
│     ├── train.csv
│     ├── tune.csv
│     └── test.csv
├── fold_2/
└── ...
```
</details>

Create a configuration file under:

- `config/training/subtyping` for training a subtyping model
- `config/training/survival` for training a survival model

You can take inspiration from `multi.yaml` files.<br>
Remember to indicate the root directory where your folds are located under `data.fold_dir`.<br>

Then, run the following command to kick off model training on multiple folds:

- subtyping: `python3 train/subtyping_multi.py --config-name <subtyping_multi_fold_config_filename>`
- survival: `python3 train/survival_multi.py --config-name <survival_single_fold_config_filename>`

## Hierarchical Pretraining

### Patch sampling
If you


<details>
<summary>
Example Directory
</summary>

```bash
PRETRAINING_DIR/
  └──patch_256_pretraining/
        └──imgs/
            ├── patch_1.png
            ├── patch_2.png
            └── ...
  └──region_4096_pretraining/
      ├── slide_1_1.pt
      ├── slide_1_2.pt
      └── ...
```
</details>

Where:
- `.../path/to/patch_256_pretraining/imgs/`: directory of raw `[256 × 256]` patches (as `*.png` format) extracted using [HS2P](https://github.com/clemsgrs/hs2p), used to pretrain the first Transformer block (ViT_patch).
- `.../path/to/region_4096_pretraining/`: directory of pre-extracted region-level **local** features for each `[4096 × 4096]` region across all WSIs using `python3 pre-train/extract_features.py`. Each `*.pt` file is a `[256 × 384]`-sized Tensor, which is a 256-length sequence of pre-extracted ViT_patch features for each `[256 × 256]` patch. This folder is used to pretain the intermediate Transformer block (ViT_region).

NB: you should be able to user differently sized regions (e.g. `[1024 × 1024]`) seamlessly.

Create two configuration files under `config/pre-training/` (one for each pre-training stage).<br>
You can take inspiration from existing files.<br>

The following commands are used for **single-gpu** pretraining:

```bash
python3 pretrain/dino_patch.py --config-name <dino_patch_config>
python3 pretrain/dino_region.py --config-name <dino_region__config>
```

**Distributed** pretraining across multiple is supported via:

```bash
python3 -m torch.distributed.run --nproc_per_node=gpu pretrain/dino_patch.py --config-name <dino_patch_config>
python3 -m torch.distributed.run --nproc_per_node=gpu pretrain/dino_region.py --config-name <dino_region_config>
```

## Resuming experiment after crash / bug

If, for some reason, feature extraction crashes, you should be able to resume from last processed slide simply by turning the `resume` parameter in your feature extraction config file to `True`, keeping all other parameters unchanged.

## TODO List

- [ ] switch back optimizer.zero_grad( ) before .step( )
- [ ] implement heatmap visualization
- [ ] add early stopping mechanism to DINO pretraining
