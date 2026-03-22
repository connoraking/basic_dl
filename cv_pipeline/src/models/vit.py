import torch
import torch.nn as nn


def get_transformer_activation(name):
    if name == "relu":
        return "relu"
    if name == "gelu":
        return "gelu"
    raise ValueError(f"Unsupported transformer activation: {name}")


class ViTClassifier(nn.Module):
    def __init__(self, model_config, num_classes, input_channels, image_size):
        super().__init__()

        if isinstance(image_size, int):
            image_height = image_size
            image_width = image_size
        else:
            image_height, image_width = image_size

        patch_size = model_config.get("patch_size", 4)
        embed_dim = model_config.get("embed_dim", 256)
        depth = model_config.get("depth", 6)
        num_heads = model_config.get("num_heads", 8)
        mlp_dim = model_config.get("mlp_dim", embed_dim * 4)
        dropout = model_config.get("dropout", 0.1)
        attention_dropout = model_config.get("attention_dropout", 0.1)
        activation = model_config.get("activation", "gelu")

        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError(
                "Image size must be divisible by patch_size. "
                f"Got image_size=({image_height}, {image_width}) and patch_size={patch_size}."
            )

        num_patches_h = image_height // patch_size
        num_patches_w = image_width // patch_size
        num_patches = num_patches_h * num_patches_w

        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation=get_transformer_activation(activation),
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(attention_dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x
