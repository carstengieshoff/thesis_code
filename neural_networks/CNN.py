from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class CNN(nn.Module):  # type: ignore
    def __init__(self, writer: Optional[SummaryWriter] = None) -> None:
        super().__init__()
        self.writer = writer
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x

    def train_nn(
        self,
        num_epochs: int,
        trainloader: DataLoader,
        validationloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[lr_scheduler.StepLR] = None,
        show_every_n: int = 100,
        plot_loss: bool = False,
        save_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Train the nn."""
        n_total_steps = len(trainloader)
        running_loss = []

        for epoch in range(num_epochs):
            self.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = labels.to(device)
                inputs = inputs.to(device)

                # forward + backward + optimize
                outputs = self.forward(inputs.float())
                loss = criterion(outputs.to(device), labels.long())
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
                            self.val_loss.append(self.evaluate_nn(validationloader, device=device))
                            self.writer.add_scalar(
                                "trainloss",
                                self.train_loss[-1],
                                epoch * n_total_steps + i,
                            )
                            self.writer.add_scalar(
                                "valacc",
                                self.val_loss[-1],
                                epoch * n_total_steps + i,
                            )
                            if save_path:
                                if (
                                    len(self.val_loss) == 1
                                    or len(self.val_loss) > 1
                                    and self.val_loss[-1] == max(self.val_loss)
                                ):
                                    torch.save(self.state_dict(), save_path)
                        else:
                            self.writer.add_scalar("trainloss", self.train_loss[-1], epoch * n_total_steps + i)

                    running_loss = []

            if scheduler is not None:
                scheduler.step()

        if save_path:
            self.load_state_dict(torch.load(save_path, map_location=torch.device(device)))

        if plot_loss:
            self.plot_loss(*args, **kwargs)

    def evaluate_nn(self, loader: DataLoader, device: torch.device) -> float:
        correct = 0
        total = 0
        self.eval()
        with torch.no_grad():
            for data in loader:
                images, labels = data
                labels = labels.to(device)
                images = images.to(device)
                # calculate outputs by running images through the network
                outputs = self.forward(images.float())
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        return 100 * correct / total

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

    def show_confusion_matrix(self, loader: DataLoader, device: torch.device, name: Optional[str] = None) -> None:
        name = "" if name is None else name
        self.eval()
        true_label: List[int] = []
        predicted_label: List[int] = []
        with torch.no_grad():
            for x, y in loader:
                y = y.to(device)
                x = x.to(device)
                outputs = self.forward(x.float().to(device))
                _, predicted = torch.max(outputs.data, 1)

                true_label = true_label + y.tolist()
                predicted_label = predicted_label + predicted.tolist()

        cm = confusion_matrix(true_label, predicted_label)
        cm = pd.DataFrame(cm)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, cmap="Blues")
        plt.xlabel("predicted")
        plt.ylabel("true")
        if self.writer is not None:
            tag = f"ConfusionMatrix_{name}"
            self.writer.add_figure(tag, fig)
        plt.show()

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":

    class Autoencoder(CNN):
        def __init__(self) -> None:
            super(Autoencoder, self).__init__()

    net = Autoencoder()

    print(net.val_loss)
