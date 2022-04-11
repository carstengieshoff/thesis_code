from typing import Dict

import torch
from torch import nn


def get_out_activations(
    model: nn.Module,
    model_input: torch.tensor,
) -> Dict[str, torch.tensor]:
    activations: Dict[str, torch.tensor] = dict()
    hooks = []

    def get_hook(name: str):  # type: ignore
        def _hook(inst: nn.Module, inp: torch.tensor, out: torch.tensor) -> None:
            activations[name] = out

        return _hook

    for name, layer in model.named_children():
        hooks.append(layer.register_forward_hook(get_hook(name)))

    model(model_input)

    for hook in hooks:
        hook.remove()

    return activations


def get_inp_activations(
    model: nn.Module,
    model_input: torch.tensor,
) -> Dict[str, torch.tensor]:

    activations: Dict[str, torch.tensor] = dict()
    hooks = []

    def get_hook(name: str):  # type: ignore
        def _hook(inst: nn.Module, inp: torch.tensor, out: torch.tensor) -> None:
            activations[name] = out

        return _hook

    for name, layer in model.named_children():
        hooks.append(layer.register_forward_hook(get_hook(name)))

    model(model_input)

    for hook in hooks:
        hook.remove()

    return activations
