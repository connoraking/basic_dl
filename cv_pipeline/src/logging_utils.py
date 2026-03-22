from pathlib import Path
import pandas as pd


def build_run_name(config):
    dataset_name = config["dataset"]["name"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    if model_cfg["name"] in {"custom_cnn", "custom_cnn_large"}:
        conv_filters = [
            str(model_cfg[key])
            for key in (
                "conv1_filters",
                "conv2_filters",
                "conv3_filters",
                "conv4_filters",
                "conv5_filters",
                "conv6_filters",
            )
            if key in model_cfg
        ]
        return (
            f"{dataset_name}_"
            f"{model_cfg['name']}_"
            f"{model_cfg['activation']}_"
            f"{'-'.join(conv_filters)}_"
            f"k{model_cfg['kernel_size']}_"
            f"drop{model_cfg['dropout']}_"
            f"{train_cfg['optimizer']}_"
            f"lr{train_cfg['learning_rate']}_"
            f"bs{train_cfg['batch_size']}_"
            f"{config['dataset']['augmentation']}"
        )

    return (
        f"{dataset_name}_"
        f"{model_cfg['name']}_"
        f"{train_cfg['optimizer']}_"
        f"lr{train_cfg['learning_rate']}_"
        f"bs{train_cfg['batch_size']}"
    )


def append_experiment_log(config, train_results, test_results):
    dataset_name = config["dataset"]["name"]
    csv_path = Path(f"outputs/logs/{dataset_name}_experiments.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = build_run_name(config)

    row = {
        "run_name": run_name,
        "dataset": dataset_name,
        "model_name": config["model"]["name"],
        "activation": config["model"].get("activation"),
        "conv1_filters": config["model"].get("conv1_filters"),
        "conv2_filters": config["model"].get("conv2_filters"),
        "conv3_filters": config["model"].get("conv3_filters"),
        "conv4_filters": config["model"].get("conv4_filters"),
        "conv5_filters": config["model"].get("conv5_filters"),
        "conv6_filters": config["model"].get("conv6_filters"),
        "kernel_size": config["model"].get("kernel_size"),
        "dropout": config["model"].get("dropout"),
        "optimizer": config["training"]["optimizer"],
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["epochs"],
        "augmentation": config["dataset"].get("augmentation"),
        "best_epoch": train_results["best_epoch"],
        "best_val_loss": train_results["best_val_loss"],
        "actual_epochs_ran": train_results["actual_epochs_ran"],
        "stopped_early": train_results["stopped_early"],
        "test_loss": test_results["test_loss"],
        "test_accuracy": test_results["test_accuracy"],
        "notes": config.get("notes", ""),
    }

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
    return run_name, csv_path
