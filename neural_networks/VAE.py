from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class VAE(nn.Module):  # type: ignore
    def __init__(self, writer: Optional[SummaryWriter] = None) -> None:
        super().__init__()
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.writer = writer
        self.kl = 0

    def encoder(self, x: torch.tensor) -> torch.tensor:
        return x, x

    def decoder(self, z: torch.tensor) -> torch.tensor:
        return z

    def reparameterize(self, mu: torch.tensor, sigma: torch.tensor) -> torch.tensor:
        std = torch.exp(sigma / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        self.kl = (std**2 + mu**2 - torch.log(std) - 1 / 2).mean()
        return z

    def forward(self, x: torch.tensor) -> torch.tensor:
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        y = self.decoder(z)
        return y

    def train_nn(
        self,
        num_epochs: int,
        trainloader: DataLoader,
        validationloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        show_every_n: int = 100,
        plot_loss: bool = False,
        lam: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Train the nn."""
        n_total_steps = len(trainloader)
        running_loss = []
        for epoch in range(num_epochs):

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                # labels = labels.to(device)
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs.float())
                loss = criterion(outputs.to(device), inputs) + lam * self.kl

                if loss.isnan().any():
                    print(inputs, labels)
                    raise RuntimeError("nans in loss")

                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss.append(loss.item())
                if i % show_every_n == (show_every_n - 1):
                    self.train_loss.append(sum(running_loss) / len(running_loss))
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {self.train_loss[-1]:.3f}")

                    if self.writer is not None:
                        if validationloader:
                            self.val_loss.append(self.evaluate_nn(validationloader, device=device, criterion=criterion))
                            self.writer.add_scalars(
                                "Training",
                                {"training loss": self.train_loss[-1], "validation loss": self.val_loss[-1]},
                                epoch * n_total_steps + i,
                            )
                        else:
                            self.writer.add_scalar("training loss", self.train_loss[-1], epoch * n_total_steps + i)

                    running_loss = []

        if plot_loss:
            self.plot_loss(*args, **kwargs)

    def evaluate_nn(self, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
        running_loss = []

        with torch.no_grad():
            for data in loader:
                images, _ = data
                images = images.to(device)

                outputs = self.forward(images.float())
                loss = criterion(outputs.to(device), images)
                running_loss.append(loss.item())

        return sum(running_loss) / len(running_loss)

    def plot_loss(self, *args: Any, **kwargs: Any) -> None:

        plt.figure(*args, **kwargs)

        ax = plt.subplot(111)
        losses = np.array(self.train_loss)
        losses_scaled = 100 * losses / losses.max()
        ax.plot(losses_scaled, label="loss")
        if self.val_loss:
            ax.plot(self.val_loss, label="validation")

        ax.legend()
        plt.show()
