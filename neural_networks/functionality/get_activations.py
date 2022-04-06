from typing import Dict

import torch
from torch import nn


def get_activations(
    model: nn.Module,
    model_input: torch.tensor,
) -> Dict[str, torch.tensor]:
    activations: Dict[str, torch.tensor] = dict()

    def _hook(inst: nn.Module, inp: torch.tensor, out: torch.tensor) -> None:
        activations[repr(inst)] = out

    hooks = []
    for i, layer in enumerate(model.children()):
        hooks.append(layer.register_forward_hook(_hook))

    model(model_input)

    for hook in hooks:
        hook.remove()

    return activations
