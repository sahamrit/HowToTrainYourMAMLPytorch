from typing import Any, Dict
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

RESNET_MODEL = 18


class ResNetClassifier(nn.Module):
    """Simple CNN for recognizing characters in a square image."""

    def __init__(
        self, data_config: Dict[str, Any], args: argparse.Namespace = None
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.num_classes = data_config["episode_classes"]

        self.resnet_model = self.args.get("resnet_model", RESNET_MODEL)

        if self.resnet_model == 18:
            self.resnet = torchvision.models.resnet18(pretrained=False)
        elif self.resnet_model == 34:
            self.resnet = torchvision.models.resnet34(pretrained=False)
        elif self.resnet_model == 50:
            self.resnet = torchvision.models.resnet50(pretrained=False)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        x = self.resnet(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--resnet_model", type=int, default=RESNET_MODEL)
        return parser
