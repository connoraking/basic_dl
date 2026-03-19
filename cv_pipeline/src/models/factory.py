from src.models.custom_cnn.baseline_3layer import CustomCNN_3layer
from src.models.resnet import ResNet18Classifier


def build_model(config, data_info):
    model_cfg = config["model"]
    name = model_cfg["name"]

    if name == "custom_cnn":
        return CustomCNN_3layer(
            model_config=model_cfg,
            num_classes=data_info["num_classes"],
            input_channels=data_info["input_channels"],
        )

    if name == "resnet18":
        return ResNet18Classifier(num_classes=data_info["num_classes"])

    raise ValueError(f"Unknown model: {name}")