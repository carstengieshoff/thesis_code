from typing import Union

import torch
from torch import nn

from neural_networks.functionality.get_activations import get_activations


def get_opt_input_for_activation(
    model: nn.Module,
    layer: Union[str, int],
    channel: int,
    device: torch.device,
    init: torch.tensor,
    num_steps: int = 1000,
) -> torch.tensor:
    init = init.float().to(device)
    init.requires_grad = True

    layer = repr(model[layer]) if isinstance(layer, int) else layer
    model.to(device)
    model.eval()

    optim = torch.optim.Adam([init], lr=1e-1, weight_decay=1e-6)

    for i in range(num_steps):
        diff = -get_activations(model=model, model_input=init)[layer][channel].mean()
        diff.backward()
        optim.step()
        optim.zero_grad()

    return init
