import os
import sys
import tqdm
import time
import json
import hydra
import wandb
import shutil
import random
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import DictConfig
from torchvision import datasets

import source.vision_transformer as vits

from source.utils import initialize_wandb, compute_time
from source.components import DINOLoss
from utils import (
    PatchDataAugmentationDINO,
    MultiCropWrapper,
    train_one_epoch,
    fix_random_seeds,
    has_batchnorms,
    get_params_groups,
    resume_from_checkpoint,
    cosine_scheduler,
    get_world_size,
    is_main_process,
    h5fileDataset,
)


@hydra.main(
    version_base="1.2.0", config_path="../config/pretraining", config_name="patch"
)
def main(cfg: DictConfig):

    distributed = torch.cuda.device_count() > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        if gpu_id == 0:
            print(f"Distributed session successfully initialized")
    else:
        gpu_id = -1
    if is_main_process():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        # set up wandb
        if cfg.wandb.enable:
            key = os.environ.get("WANDB_API_KEY")
            wandb_run = initialize_wandb(cfg, key=key)
            wandb_run.define_metric("epoch", summary="max")

    fix_random_seeds(cfg.seed)

    cudnn.benchmark = True

    output_dir = Path(cfg.output_dir, cfg.experiment_name)
    if not cfg.resume and is_main_process():
        if output_dir.exists():
            print(
                f"WARNING: {output_dir} already exists! Deleting its content...")
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    # preparing data
    if is_main_process():
        print(f"Loading data...")

    transform = PatchDataAugmentationDINO(
        cfg.aug.global_crops_scale,
        cfg.aug.local_crops_scale,
        cfg.aug.local_crops_number,
    )

    if cfg.extension == "jpeg":
        dataset = datasets.ImageFolder(cfg.data_dir, transform=transform)
    elif cfg.extension == "h5":
        dataset = h5fileDataset(cfg.data_dir, transform=transform)

        
    if cfg.training.pct:
        print(f"Pre-training on {cfg.training.pct*100}% of the data")
        nsample = int(cfg.training.pct * len(dataset))
        idxs = random.sample(range(len(dataset)), k=nsample)
        dataset = torch.utils.data.Subset(dataset, idxs)

    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.training.batch_size_per_gpu,
        num_workers=cfg.speed.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if is_main_process():
        print(f"Data loaded: there are {len(dataset)} patches.")
        print(f'Batch size per gpu is {cfg.training.batch_size_per_gpu}')

    # building student and teacher networks
    if is_main_process():
        print(f"Building student and teacher networks...")
    student = vits.__dict__[cfg.model.arch](
        patch_size=cfg.model.patch_size,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    teacher = vits.__dict__[cfg.model.arch](patch_size=cfg.model.patch_size)
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(
        student,
        vits.DINOHead(
            embed_dim,
            cfg.model.out_dim,
            use_bn=cfg.model.use_bn_in_head,
            norm_last_layer=cfg.model.norm_last_layer,
        ),
    )
    teacher = MultiCropWrapper(
        teacher,
        vits.DINOHead(
            embed_dim,
            cfg.model.out_dim,
            use_bn=cfg.model.use_bn_in_head,
        ),
    )

    # move networks to gpu
    if distributed:
        student, teacher = student.to(gpu_id), teacher.to(gpu_id)
    else:
        student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if has_batchnorms(student) and distributed:
        # we need DDP wrapper to have synchro batch norms working...
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[gpu_id], output_device=gpu_id)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    if distributed:
        student = nn.parallel.DistributedDataParallel(
            student, device_ids=[gpu_id], output_device=gpu_id)

    # teacher and student start with the same weights
    student_sd = student.state_dict()
    nn.modules.utils.consume_prefix_in_state_dict_if_present(
        student_sd, "module.")
    teacher_without_ddp.load_state_dict(student_sd)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # total number of crops = 2 global crops + local_crops_number
    crops_number = cfg.aug.local_crops_number + 2
    dino_loss = DINOLoss(
        cfg.model.out_dim,
        crops_number,
        cfg.model.warmup_teacher_temp,
        cfg.model.teacher_temp,
        cfg.model.warmup_teacher_temp_epochs,
        cfg.training.nepochs,
    )
    if distributed:
        dino_loss = dino_loss.to(gpu_id)
    else:
        dino_loss = dino_loss.cuda()

    params_groups = get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)

    # for mixed precision training
    fp16_scaler = None
    if cfg.speed.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    assert (
        cfg.training.nepochs >= cfg.training.warmup_epochs
    ), f"nepochs ({cfg.training.nepochs}) must be greater than or equal to warmup_epochs ({cfg.training.warmup_epochs})"
    base_lr = (
        cfg.optim.lr * (cfg.training.batch_size_per_gpu *
                        get_world_size()) / 256.0
    )
    lr_schedule = cosine_scheduler(
        base_lr,
        cfg.optim.lr_scheduler.min_lr,
        cfg.training.nepochs,
        len(data_loader),
        warmup_epochs=cfg.training.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        cfg.optim.lr_scheduler.weight_decay,
        cfg.optim.lr_scheduler.weight_decay_end,
        cfg.training.nepochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        cfg.model.momentum_teacher, 1, cfg.training.nepochs, len(data_loader)
    )
    if is_main_process():
        print(f"Models built, kicking off training")

    epochs_run = 0
    
    # leverage torch native fault tolerance
    snapshot_path = Path(output_dir, "latest.pt")
    if distributed:
        if snapshot_path.exists():
            print("Loading snapshot")
            loc = f"cuda:{gpu_id}"
            snapshot = torch.load(snapshot_path, map_location=loc)
            epochs_run = snapshot["epoch"]
            student.load_state_dict(snapshot["student"])
            teacher.load_state_dict(snapshot["teacher"])
            optimizer.load_state_dict(snapshot["optimizer"])
            dino_loss.load_state_dict(snapshot["dino_loss"])
            if fp16_scaler is not None:
                fp16_scaler.load_state_dict(snapshot["fp16_scaler"])
            print(f"Resuming training from snapshot at Epoch {epochs_run}")
        elif cfg.finetune:
            ckpt_path = Path(cfg.finetune_from_checkpoint)
            print('Loading checkpoint for finetuning')
            loc = f"cuda:{gpu_id}"
            snapshot = torch.load(ckpt_path, map_location=loc)
            epochs_ = snapshot["epoch"]
            student.load_state_dict(snapshot["student"])
            teacher.load_state_dict(snapshot["teacher"])
            dino_loss.load_state_dict(snapshot["dino_loss"])
            if fp16_scaler is not None:
                fp16_scaler.load_state_dict(snapshot["fp16_scaler"])
            print(
                f"Finetuning for {cfg.training.nepochs} epochs from checkpoint that already ran {epochs_} epochs")
    elif cfg.resume:
        ckpt_path = Path(output_dir, cfg.resume_from_checkpoint)
        epochs_run = resume_from_checkpoint(
            ckpt_path,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
        print(f"Resuming training from checkpoint at Epoch {epochs_run}")
    elif cfg.finetune:
        ckpt_path = Path(cfg.finetune_from_checkpoint)
        epochs_ = resume_from_checkpoint(
            ckpt_path,
            student=student,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
        epochs_run = 0
        print(
            f"Finetuning for {cfg.training.nepochs} epochs from checkpoint that already ran {epochs_} epochs")

    start_time = time.time()

    with tqdm.tqdm(
        range(epochs_run, cfg.training.nepochs),
        desc=(f"DINO Pre-Training"),
        unit=" epoch",
        ncols=100,
        leave=True,
        initial=epochs_run,
        total=cfg.training.nepochs,
        file=sys.stdout,
    ) as t:

        for epoch in t:

            epoch_start_time = time.time()
            if cfg.wandb.enable and is_main_process():
                wandb.log({"epoch": epoch + 1})

            if distributed:
                data_loader.sampler.set_epoch(epoch)

            # training one epoch of DINO
            train_stats = train_one_epoch(
                student,
                teacher,
                teacher_without_ddp,
                dino_loss,
                data_loader,
                optimizer,
                lr_schedule,
                wd_schedule,
                momentum_schedule,
                epoch,
                cfg.training.nepochs,
                fp16_scaler,
                cfg.training.clip_grad,
                cfg.training.freeze_last_layer,
                gpu_id,
                output_dir,
                cfg.wandb.enable,
            )

            lr = train_stats["lr"]
            loss = train_stats["loss"]
            if cfg.wandb.enable and is_main_process():
                wandb.define_metric("lr", step_metric="epoch")
                wandb.define_metric("loss", step_metric="epoch")
                wandb.log({"lr": lr})
                wandb.log({"loss": loss})

            # save snapshot
            if is_main_process():
                snapshot = {
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "dino_loss": dino_loss.state_dict(),
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                }
                if fp16_scaler is not None:
                    snapshot["fp16_scaler"] = fp16_scaler.state_dict()
                torch.save(snapshot, snapshot_path)

                if (
                    cfg.logging.save_snapshot_every
                    and epoch % cfg.logging.save_snapshot_every == 0
                    and is_main_process()
                ):
                    save_path = Path(
                        output_dir, f"snapshot_epoch_{epoch:03}.pth")
                    torch.save(snapshot, save_path)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            if is_main_process():
                with open(Path(output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(
                epoch_start_time, epoch_end_time)
            if is_main_process():
                tqdm.tqdm.write(
                    f"End of epoch {epoch+1}/{cfg.training.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s \t LR: {lr:.6f} \t Loss: {loss:.6f}",
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    # saving the final model to onnx
    if is_main_process():
        print('Saving the final model to onnx')
        if distributed:
            torch.onnx.export(teacher, torch.randn(
                2, 3, 224, 224, device=f"cuda:{gpu_id}"), "teacher_final_epoch.onnx")
        else:
            torch.onnx.export(teacher, torch.randn(
                2, 3, 224, 224, device="cuda"), "teacher_final_epoch.onnx")
        wandb.save("teacher_final_epoch.onnx")

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    # python3 -m torch.distributed.run pretrain/dino_patch.py --config-name 'patch'
    # python3 -m torch.distributed.run --nproc_per_node=gpu pretrain/dino_patch.py --config-name 'patch'

    # ISSUE WITH TORCHRUN ON SOL2: USES PYTHON3.8 INSTEAD OF PYTHON3.9 FOR SOME REASON
    # torchrun --standalone pretrain/dino_patch.py --nproc_per_node=gpu --config-name 'debug'

    # m = {}
    # for i in range(torch.cuda.device_count()):
    #     m_i = {f"--local_rank={i}": "local_rank"}
    #     m.update(m_i)
    # hydra_argv_remapper(m)

    main()
