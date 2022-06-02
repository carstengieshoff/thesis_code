from typing import Optional

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from neural_networks.CNN import CNN


def squeeze_weights(m: nn.Module) -> None:
    m.weight.data = m.weight.data.mean(dim=1)[:, None]
    m.in_channels = 1


class VGG(CNN):
    def __init__(self, num_out: int, writer: Optional[SummaryWriter] = None, pretrained: bool = True):
        super().__init__(writer)
        vgg = torchvision.models.vgg19(pretrained=pretrained)
        vgg.features[0].apply(squeeze_weights)

        # for i, layer in enumerate(an.features):
        #  if isinstance(layer, nn.ReLU):
        #    an.features[i] = nn.ReLU(inplace=False)

        num_ftrs = vgg.classifier[0].in_features
        self.features = vgg.features

        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_out),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x
