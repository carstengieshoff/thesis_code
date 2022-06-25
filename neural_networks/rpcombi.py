import torch
from torch import nn


class RPCombi(nn.Module):  # type: ignore
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, intermediate_channels: int = 6
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.intermediate_channels = intermediate_channels
        self._thresh_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.intermediate_channels,
            kernel_size=self.kernel_size,
            padding="same",
        )

        self._1by1conv = nn.Conv2d(
            in_channels=self.intermediate_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            padding="same",
        )

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(intermediate_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self._thresh_layer(x)
        x = self.activation1(x)
        x = self.bn1(x)
        x = self._1by1conv(x)
        x = self.activation2(x)
        return x
