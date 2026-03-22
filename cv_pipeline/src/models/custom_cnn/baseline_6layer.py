import torch.nn as nn

from src.models.custom_cnn.baseline_3layer import get_activation


class CustomCNN_6layer(nn.Module):
    def __init__(self, model_config, num_classes, input_channels=3):
        super().__init__()

        act_name = model_config["activation"]
        c1 = model_config["conv1_filters"]
        c2 = model_config["conv2_filters"]
        c3 = model_config["conv3_filters"]
        c4 = model_config["conv4_filters"]
        c5 = model_config["conv5_filters"]
        c6 = model_config["conv6_filters"]
        k = model_config["kernel_size"]
        padding = k // 2
        hidden_dim = model_config.get("classifier_hidden_dim", 512)
        dropout = model_config.get("dropout", 0.0)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=k, padding=padding),
            nn.BatchNorm2d(c1),
            get_activation(act_name),
            nn.Conv2d(c1, c2, kernel_size=k, padding=padding),
            nn.BatchNorm2d(c2),
            get_activation(act_name),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=k, padding=padding),
            nn.BatchNorm2d(c3),
            get_activation(act_name),
            nn.Conv2d(c3, c4, kernel_size=k, padding=padding),
            nn.BatchNorm2d(c4),
            get_activation(act_name),
            nn.MaxPool2d(2),
            nn.Conv2d(c4, c5, kernel_size=k, padding=padding),
            nn.BatchNorm2d(c5),
            get_activation(act_name),
            nn.Conv2d(c5, c6, kernel_size=k, padding=padding),
            nn.BatchNorm2d(c6),
            get_activation(act_name),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c6, hidden_dim),
            get_activation(act_name),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
