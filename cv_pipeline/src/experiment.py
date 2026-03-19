import os
import random
from pathlib import Path

import numpy as np
import yaml
import torch

from src.data import get_dataset
from src.models.factory import build_model
from src.trainer import train_model, test_model
from src.logging_utils import append_experiment_log, build_run_name


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def configure_reproducibility(config):
    train_cfg = config.get("training", {})
    seed = train_cfg.get("seed", 42)
    deterministic = train_cfg.get("deterministic", True)
    warn_only = train_cfg.get("deterministic_warn_only", False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Needed for deterministic CUDA matmul kernels on supported setups.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    return {
        "seed": seed,
        "deterministic": deterministic,
        "deterministic_warn_only": warn_only,
    }


def run_experiment(config_or_path):
    if isinstance(config_or_path, str):
        config = load_config(config_or_path)
    else:
        config = config_or_path

    reproducibility = configure_reproducibility(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(
        "Reproducibility settings: "
        f"seed={reproducibility['seed']}, "
        f"deterministic={reproducibility['deterministic']}, "
        f"warn_only={reproducibility['deterministic_warn_only']}"
    )

    data_info = get_dataset(config)

    model = build_model(config, data_info).to(device)

    run_name = build_run_name(config)
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"

    train_results = train_model(
        config=config,
        model=model,
        train_loader=data_info["train_loader"],
        val_loader=data_info["val_loader"],
        device=device,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
    )

    best_model = build_model(config, data_info).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_model.load_state_dict(checkpoint["model_state_dict"])

    test_results = test_model(
        model=best_model,
        test_loader=data_info["test_loader"],
        device=device,
    )

    run_name, csv_path = append_experiment_log(config, train_results, test_results)

    results = {
        "run_name": run_name,
        "dataset": data_info["dataset_name"],
        "best_epoch": train_results["best_epoch"],
        "best_val_loss": train_results["best_val_loss"],
        "test_loss": test_results["test_loss"],
        "test_accuracy": test_results["test_accuracy"],
        "seed": reproducibility["seed"],
        "deterministic": reproducibility["deterministic"],
        "log_path": str(csv_path),
        "checkpoint_path": str(checkpoint_path),
    }

    print(results)
    return results
