from typing import Optional

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def save_weights(
    writer: SummaryWriter, model: nn.Module, global_name: Optional[str] = None, layer_name: str = "conv"
) -> None:

    global_name = "" if global_name is None else global_name + "_"

    for layer in model.children():
        if repr(layer).lower().startswith(layer_name):
            weight = layer.weight.detach()
            weight = weight.reshape(-1, 1, weight.shape[-2], weight.shape[-1])

            img_grid = make_grid(weight)
            writer.add_image(global_name + layer, img_grid)
