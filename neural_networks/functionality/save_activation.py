from typing import List, Optional, Union

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from neural_networks.functionality.get_activations import get_activations


def save_activation(
    writer: SummaryWriter,
    model: nn.Module,
    model_input: torch.tensor,
    layers: List[Union[str, int]],
    global_name: Optional[str] = None,
) -> None:

    if not isinstance(model, nn.Sequential):
        raise RuntimeError(f"Expecting `model` to be of type `nn.Sequential`, got {type(model)}")

    global_name = "" if global_name is None else global_name + "_"
    activations = get_activations(model=model, model_input=model_input)

    for layer in layers:
        if isinstance(layer, int):
            layer = repr(model[layer])

        activation = activations[layer]
        N, C, H, W = activation.shape

        for i in range(N):
            img_grid = make_grid(activation[i].reshape(-1, 1, H, W))
            tag = global_name + layer + f"_{i}"
            writer.add_image(tag, img_grid)
