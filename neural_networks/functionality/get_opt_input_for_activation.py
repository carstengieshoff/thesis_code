import torch
from torch import nn

from neural_networks.functionality.get_activations import get_out_activations


def get_opt_input_for_activation(
    model: nn.Module,
    layer_name: str,
    channel: int,
    device: torch.device,
    init: torch.tensor,
    num_steps: int = 1000,
) -> torch.tensor:
    init = init.float().to(device)
    init.requires_grad = True

    model.to(device)
    model.eval()

    optim = torch.optim.Adam([init], lr=1e-1, weight_decay=1e-6)

    for i in range(num_steps):
        activation = get_out_activations(model=model, model_input=init)[layer_name]
        diff = -1 * activation[:, channel].mean()
        diff.backward()
        optim.step()
        optim.zero_grad()

    return init
