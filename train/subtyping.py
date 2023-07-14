import os
import time
import tqdm
import wandb
import torch
import hydra
import datetime
import pandas as pd

from pathlib import Path
from functools import partial
from omegaconf import DictConfig

from source.models import ModelFactory
from source.components import LossFactory
from source.dataset import ExtractedFeaturesDataset
from source.utils import (
    fix_random_seeds,
    initialize_wandb,
    train,
    tune,
    test,
    compute_time,
    update_log_dict,
    collate_features,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)


@hydra.main(
    version_base="1.2.0",
    config_path="../config/training/classification",
    config_name="single",
)
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        run_id = wandb_run.id

    fix_random_seeds(0)
    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(output_dir, "checkpoints", cfg.level)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(output_dir, "results", cfg.level)
    result_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cfg.features_dir)

    num_classes = cfg.num_classes
    criterion = LossFactory(cfg.task, cfg.loss, cfg.loss_options).get_loss()

    model = ModelFactory(cfg.level, num_classes, cfg.task, cfg.label_mapping, cfg.model).get_model()
    model.relocate()
    print(model)

    print(f"Loading data")
    train_df = pd.read_csv(cfg.data.train_csv)
    tune_df = pd.read_csv(cfg.data.tune_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    if cfg.training.pct:
        print(f"Training & Tuning on {cfg.training.pct*100}% of the data")
        train_df = train_df.sample(frac=cfg.training.pct).reset_index(drop=True)
        tune_df = tune_df.sample(frac=cfg.training.pct).reset_index(drop=True)

    print(f"Initializing training dataset")
    train_dataset = ExtractedFeaturesDataset(
        train_df,
        features_dir,
        cfg.label_name,
        cfg.label_mapping,
    )
    print(f"Initializing tuning dataset")
    tune_dataset = ExtractedFeaturesDataset(
        tune_df,
        features_dir,
        cfg.label_name,
        cfg.label_mapping,
    )
    print(f"Initializing testing dataset")
    test_dataset = ExtractedFeaturesDataset(
        test_df,
        features_dir,
        cfg.label_name,
        cfg.label_mapping,
    )

    m, n = train_dataset.num_classes, tune_dataset.num_classes
    assert (
        m == n == cfg.num_classes
    ), f"Either train (C={m}) or tune (C={n}) sets doesnt cover full class spectrum (C={cfg.num_classes}"

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OptimizerFactory(
        cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
    ).get_optimizer()
    scheduler = SchedulerFactory(optimizer, cfg.optim.lr_scheduler).get_scheduler()

    early_stopping = EarlyStopping(
        cfg.early_stopping.tracking,
        cfg.early_stopping.min_max,
        cfg.early_stopping.patience,
        cfg.early_stopping.min_epoch,
        checkpoint_dir=checkpoint_dir,
        save_all=cfg.early_stopping.save_all,
    )

    stop = False
    start_time = time.time()

    with tqdm.tqdm(
        range(cfg.nepochs),
        desc=(f"HIPT Training"),
        unit=" slide",
        ncols=100,
        leave=True,
    ) as t:

        for epoch in t:

            epoch_start_time = time.time()
            if cfg.wandb.enable:
                log_dict = {"epoch": epoch + 1}

            train_results = train(
                epoch + 1,
                model,
                train_dataset,
                optimizer,
                criterion,
                collate_fn=partial(collate_features, label_type="int"),
                batch_size=cfg.training.batch_size,
                weighted_sampling=cfg.training.weighted_sampling,
                gradient_accumulation=cfg.training.gradient_accumulation,
            )

            if cfg.wandb.enable:
                update_log_dict(
                    "train", train_results, log_dict, to_log=cfg.wandb.to_log
                )
            train_dataset.df.to_csv(Path(result_dir, f"train_{epoch}.csv"), index=False)

            if epoch % cfg.tuning.tune_every == 0:

                tune_results = tune(
                    epoch + 1,
                    model,
                    tune_dataset,
                    criterion,
                    collate_fn=partial(collate_features, label_type="int"),
                    batch_size=cfg.tuning.batch_size,
                )

                if cfg.wandb.enable:
                    update_log_dict(
                        "tune", tune_results, log_dict, to_log=cfg.wandb.to_log
                    )
                tune_dataset.df.to_csv(
                    Path(result_dir, f"tune_{epoch}.csv"), index=False
                )

                early_stopping(epoch, model, tune_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            lr = cfg.optim.lr
            if scheduler:
                lr = scheduler.get_last_lr()[0]
                scheduler.step()
            if cfg.wandb.enable:
                log_dict.update({"train/lr": lr})

            # logging
            if cfg.wandb.enable:
                wandb.log(log_dict, step=epoch + 1)

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            tqdm.tqdm.write(
                f"End of epoch {epoch+1} / {cfg.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
            )

            if stop:
                tqdm.tqdm.write(
                    f"Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago"
                )
                break

    # load best model
    best_model_fp = Path(checkpoint_dir, f"{cfg.testing.retrieve_checkpoint}_model.pt")
    if cfg.wandb.enable:
        wandb.save(str(best_model_fp))
    best_model_sd = torch.load(best_model_fp)
    model.load_state_dict(best_model_sd)

    test_results = test(
        model,
        test_dataset,
        collate_fn=partial(collate_features, label_type="int"),
        batch_size=1,
    )
    test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)

    for r, v in test_results.items():
        if r == "auc":
            v = round(v, 3)
        if r in cfg.wandb.to_log and cfg.wandb.enable:
            wandb.log({f"test/{r}": v})

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    main()
