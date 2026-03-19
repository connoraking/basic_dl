import torch
import torch.nn as nn


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class CustomCNN_3layer(nn.Module):
    def __init__(self, model_config, num_classes, input_channels=3):
        super().__init__()

        act_name = model_config["activation"]
        c1 = model_config["conv1_filters"]
        c2 = model_config["conv2_filters"]
        c3 = model_config["conv3_filters"]
        k = model_config["kernel_size"]
        hidden_dim = model_config.get("classifier_hidden_dim", 256)

        self.conv1 = nn.Conv2d(input_channels, c1, kernel_size=k, padding=1)
        self.act1 = get_activation(act_name)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=k, padding=1)
        self.act2 = get_activation(act_name)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=k, padding=1)
        self.act3 = get_activation(act_name)
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(c3 * 4 * 4, hidden_dim)
        self.act4 = get_activation(act_name)
        self.dropout = nn.Dropout(model_config.get("dropout", 0.0))
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))

        x = self.flatten(x)
        x = self.act4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x