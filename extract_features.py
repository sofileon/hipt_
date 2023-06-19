import os
import h5py
import tqdm
import time
import wandb
import torch
import hydra
import shutil
import datetime
import patchify
import subprocess
import pandas as pd
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from torchvision import transforms

from source.dataset import RegionFilepathsDataset, RegionFilepathsDatasetv2
from source.models import GlobalFeatureExtractor, LocalFeatureExtractor
from source.utils import initialize_wandb, initialize_df, collate_region_filepaths, is_main_process


@hydra.main(
    version_base="1.2.0", config_path="config/feature_extraction", config_name="default"
)
def main(cfg: DictConfig):

    distributed = torch.cuda.device_count() > 1 and cfg.multi
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        if gpu_id == 0:
            print(f"Distributed session successfully initialized")
    else:
        gpu_id = -1

    if is_main_process():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        # set up wandb
        if cfg.wandb.enable:
            key = os.environ.get("WANDB_API_KEY")
            wandb_run = initialize_wandb(cfg, key=key)
            wandb_run.define_metric("processed", summary="max")
            run_id = wandb_run.id
    else:
        run_id = ""

    if distributed:
        obj = [run_id]
        torch.distributed.broadcast_object_list(obj, 0, device=torch.device(f"cuda:{gpu_id}"))
        run_id = obj[0]

    output_dir = Path(cfg.output_dir, cfg.experiment_name)#, run_id)
    slide_features_dir = Path(output_dir, "slide")
    region_features_dir = Path(output_dir, "region")
    #check if output dir exists if not create it
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        slide_features_dir.mkdir()
        if cfg.save_region_features:
            region_features_dir.mkdir()
    
    if not cfg.resume and is_main_process():
        if output_dir.exists():
            print(f"{output_dir} already exists! deleting it...")
            shutil.rmtree(output_dir)
            print("done")
            output_dir.mkdir(parents=False)
            slide_features_dir.mkdir()
            if cfg.save_region_features:
                region_features_dir.mkdir()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            slide_features_dir.mkdir(exist_ok=True)
            if cfg.save_region_features:
                region_features_dir.mkdir(exist_ok=True)

    if cfg.level == "global":
        model = GlobalFeatureExtractor(
            region_size=cfg.region_size,
            patch_size=cfg.patch_size,
            mini_patch_size=cfg.mini_patch_size,
            pretrain_vit_patch=cfg.pretrain_vit_patch,
            pretrain_vit_region=cfg.pretrain_vit_region,
            verbose=(gpu_id in [-1, 0]),
        )
    elif cfg.level == "local":
        model = LocalFeatureExtractor(
            patch_size=cfg.patch_size,
            mini_patch_size=cfg.mini_patch_size,
            pretrain_vit_patch=cfg.pretrain_vit_patch,
            verbose=(gpu_id in [-1, 0]),
        )
    else:
        raise ValueError(f"cfg.level ({cfg.level}) not supported")

    region_dir = Path(cfg.region_dir)
    if cfg.format == "h5":
        slide_ids = sorted([s.name for sub in region_dir.iterdir()
                        for s in sub.iterdir()])
    else:
        slide_ids = sorted([s.name for s in region_dir.iterdir()])
    if is_main_process():
        print(f"{len(slide_ids)} slides with extracted patches found")

    if cfg.slide_list:
        with open(Path(cfg.slide_list), "r") as f:
            slide_ids = sorted([Path(x.strip()).stem for x in f.readlines()])
            if cfg.resume:
                #check the slides from the slide_list that have already been processed
                processed_slides = [slide.stem for slide in slide_features_dir.iterdir() if slide.stem in slide_ids]
                slide_ids = [slide for slide in slide_ids if slide not in processed_slides] #keep only the slides that have not been processed
        if is_main_process():
            if not cfg.resume:
                print(f"restricting to {len(slide_ids)} slides from slide list .txt file")
            else:
                print(
                    f"restricting to {len(slide_ids)}/{len(slide_ids)+len(processed_slides)} slides from slide list .txt file ({len(processed_slides)} already processed)")

    df = initialize_df(slide_ids)
    if cfg.format == "h5":
        dataset = RegionFilepathsDatasetv2(df, region_dir, cfg.format)
    else:
        dataset = RegionFilepathsDataset(df, region_dir, cfg.format)

    if distributed and is_main_process() and cfg.wandb.enable:
        command_line = [
            "python3",
            "log_nproc.py",
            "--output_dir",
            f"{slide_features_dir}",
            "--fmt",
            "pt",
            "--total",
            f"{len(dataset)}",
            "--log_to_wandb",
            "--id",
            f"{run_id}"
        ]
        subprocess.Popen(command_line)

    time.sleep(5)

    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_region_filepaths,
    )

    if gpu_id == -1:
        device = torch.device(f"cuda")
    else:
        device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device, non_blocking=True)

    if is_main_process():
        print()

    slide_ids, region_slide_ids = [], []
    slide_feature_paths, region_feature_paths = [], []
    x_coords, y_coords = [], []

    with tqdm.tqdm(
        loader,
        desc="Slide Encoding",
        unit=" slide",
        ncols=80,
        position=0,
        leave=True,
        disable=not (gpu_id in [-1, 0]),
    ) as t1:

        with torch.no_grad():

            for i, batch in enumerate(t1):

                idx, region_fps, slide_id = batch
                slide_ids.append(slide_id)
                features = []

                with tqdm.tqdm(
                    region_fps,
                    desc=(f"{slide_id}"),
                    unit=" region",
                    ncols=80 + len(slide_id),
                    position=1,
                    leave=False,
                    disable=not (gpu_id in [-1, 0]),
                ) as t2:

                    for fp in t2:

                        x_y = Path(fp).stem
                        if cfg.format == "h5":
                            x, y = int(x_y.split('_')[-2]), int(x_y.split('_')[-1])
                            img = h5py.File(fp, 'r')['imgs'][:].reshape(16, 16, 1, 256, 256, 3)
                            img = Image.fromarray(patchify.unpatchify(
                                img, (cfg.region_size, cfg.region_size, 3)))
                        else:
                            x, y = int(x_y.split('_')[0]), int(x_y.split('_')[1])
                            img = Image.open(fp)
                        x_coords.append(x)
                        y_coords.append(y)
                        img = transforms.functional.to_tensor(img)  # [3, 4096, 4096]
                        img = img.unsqueeze(0)  # [1, 3, 4096, 4096]
                        img = img.to(device, non_blocking=True)
                        feature = model(img)
                        if cfg.save_region_features:
                            save_path = Path(
                                region_features_dir, f"{slide_id}_{x}_{y}.pt")
                            torch.save(feature, save_path)
                            region_feature_paths.append(save_path.resolve())
                            region_slide_ids.append(slide_id)
                        features.append(feature)

                stacked_features = torch.stack(features, dim=0).squeeze(1)
                save_path = Path(slide_features_dir, f"{slide_id}.pt")
                torch.save(stacked_features, save_path)
                slide_feature_paths.append(save_path.resolve())

                if cfg.wandb.enable and not distributed:
                    wandb.log({"processed": i + 1})

    slide_features_df = pd.DataFrame.from_dict({
        'feature_path': slide_feature_paths,
        'slide_id': slide_ids,
        'level': [f'{cfg.level}']*len(slide_ids),
        'tile_size': [cfg.region_size]*len(slide_ids),
    })

    if distributed:
        slide_features_csv_path = Path(
            output_dir, f'slide_features_{gpu_id}.csv')
    else:
        slide_features_csv_path = Path(output_dir, f'slide_features.csv')
    slide_features_df.to_csv(slide_features_csv_path, index=False)

    if cfg.save_region_features:
        region_features_df = pd.DataFrame.from_dict({
            'feature_path': region_feature_paths,
            'slide_id': region_slide_ids,
            'level': [f'{cfg.level}']*len(region_slide_ids),
            'tile_size': [cfg.region_size]*len(region_slide_ids),
            'x': x_coords,
            'y': y_coords,
        })
        if distributed:
            region_features_csv_path = Path(
                output_dir, f'region_features_{gpu_id}.csv')
        else:
            region_features_csv_path = Path(output_dir, f'region_features.csv')
        region_features_df.to_csv(region_features_csv_path, index=False)

    if distributed:
        torch.distributed.barrier()
        if is_main_process():
            slide_dfs = []
            if cfg.save_region_features:
                region_dfs = []
            for gpu_id in range(torch.cuda.device_count()):
                slide_fp = Path(output_dir, f'slide_features_{gpu_id}.csv')
                slide_df = pd.read_csv(slide_fp)
                slide_dfs.append(slide_df)
                os.remove(slide_fp)
                if cfg.save_region_features:
                    region_fp = Path(
                        output_dir, f'region_features_{gpu_id}.csv')
                    region_df = pd.read_csv(region_fp)
                    region_dfs.append(region_df)
                    os.remove(region_fp)
            slide_features_df = pd.concat(slide_dfs, ignore_index=True)
            slide_features_df = slide_features_df.drop_duplicates()
            slide_features_df.to_csv(
                Path(output_dir, f'slide_features.csv'), index=False)
            if cfg.save_region_features:
                region_features_df = pd.concat(region_dfs, ignore_index=True)
                region_features_df = region_features_df.drop_duplicates()
                region_features_df.to_csv(
                    Path(output_dir, f'region_features.csv'), index=False)

    if cfg.wandb.enable and is_main_process() and distributed:
        wandb.log({"processed": len(slide_features_df)})


if __name__ == "__main__":

    # python3 -m torch.distributed.run --standalone --nproc_per_node=gpu extract_features_dist.py --config-name 'HIPT_tcga_ckpoint_local'
    main()
