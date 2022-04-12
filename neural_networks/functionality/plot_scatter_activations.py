from typing import Any, List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from neural_networks.functionality.get_activations import get_inp_activations


def plot_scatter(
    model: nn.Module,
    loader: DataLoader,
    layer_name: str,
    device: torch.device,
    dim_reduction: str = "normal",
    writer: Optional[SummaryWriter] = None,
    *args: Any,
    **kwargs: Any,
) -> None:

    vectors: List[torch.tensor] = []
    labels: List[torch.tensor] = []

    for input, label in loader:
        input = input.to(device)
        vectors.append(get_inp_activations(model=model, model_input=input)[layer_name])
        labels.append(label)

    vectors_np: np.array = torch.cat(vectors).detach().cpu().numpy()
    labels_np: np.array = torch.cat(labels).detach().cpu().numpy()

    if dim_reduction == "normal":
        X = vectors_np[:, :2]
    elif dim_reduction == "pca":
        transformer = PCA(n_components=2)
        X = transformer.fit_transform(vectors_np)
    elif dim_reduction == "tsne":
        transformer = TSNE(n_components=2)
        X = transformer.fit_transform(vectors_np)
    else:
        raise KeyError(f"`dim_reduction` is expected to be one of `normal`, `pca`, `tsne`, got {dim_reduction}")

    fig = plt.figure(*args, **kwargs)
    plt.scatter(X[:, 0], X[:, 1], c=labels_np)
    plt.legend()

    if writer is not None:
        writer.add_figure(tag=f"Scatter_{layer_name}", fig=fig)

    plt.show()
