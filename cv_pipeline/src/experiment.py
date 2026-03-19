from pathlib import Path
import yaml
import torch

from src.data import get_dataset
from src.models.factory import build_model
from src.trainer import train_model, test_model
from src.logging_utils import append_experiment_log, build_run_name


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config_path):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

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
        "log_path": str(csv_path),
        "checkpoint_path": str(checkpoint_path),
    }

    print(results)
    return results