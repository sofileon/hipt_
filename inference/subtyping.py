import os
import time
import wandb
import hydra
import torch
import datetime
import pandas as pd

from pathlib import Path
from functools import partial
from omegaconf import DictConfig

from source.models import ModelFactory
from source.dataset import ExtractedFeaturesDataset
from source.utils import (
    initialize_wandb,
    test,
    compute_time,
    collate_features,
)


@hydra.main(
    version_base="1.2.0",
    config_path="../config/inference/subtyping",
    config_name="default",
)
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(output_dir, "results", cfg.level)
    result_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cfg.features_dir)

    num_classes = cfg.num_classes

    model = ModelFactory(cfg.level, num_classes, cfg.task, model_options=cfg.model).get_model()
    model.relocate()
    print(model)

    print(f"Loading test data")
    test_df = pd.read_csv(cfg.test_csv)

    print(f"Initializing test dataset")
    test_dataset = ExtractedFeaturesDataset(
        test_df,
        features_dir,
        cfg.label_name,
        cfg.label_mapping,
    )

    print(f"Loading provided model checkpoint")
    sd = torch.load(cfg.model.checkpoint)
    msg = model.load_state_dict(sd)
    print(f"Checkpoint loaded with msg: {msg}")

    print(f"Running inference on test dataset")
    start_time = time.time()

    test_results = test(
        model,
        test_dataset,
        collate_fn=partial(collate_features, label_type="int"),
        batch_size=1,
    )
    test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)
    print()

    for r, v in test_results.items():
        if r == "auc":
            v = round(v, 3)
        if r in cfg.wandb.to_log and cfg.wandb.enable:
            wandb.log({f"test/{r}": v})
        else:
            print(f"Test {r}: {v}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    main()
