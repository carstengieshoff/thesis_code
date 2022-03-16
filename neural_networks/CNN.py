from typing import Any, List

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


class CNN(nn.Module):  # type: ignore
    def __init__(self) -> None:
        super().__init__()
        self.losses: List[float] = []
        self.val_acc: List[float] = []

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
        show_every_n: int = 100,
        plot_loss: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Train the nn."""
        running_loss = []
        for epoch in range(num_epochs):

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = labels.to(device)
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs.float())
                loss = criterion(outputs.to(device), labels.long())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss.append(loss.item())
                if i % show_every_n == (show_every_n - 1):
                    self.losses.append(sum(running_loss) / len(running_loss))
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {self.losses[-1]:.3f}")
                    running_loss = []
                    if validationloader:
                        self.val_acc.append(self.evaluate_nn(validationloader, device=device))

        if plot_loss:
            self.plot_loss(*args, **kwargs)

    def evaluate_nn(self, loader: DataLoader, device: torch.device) -> float:
        correct = 0
        total = 0

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

        return 100 * correct // total

    def plot_loss(self, *args: Any, **kwargs: Any) -> None:

        plt.figure(*args, **kwargs)

        ax = plt.subplot(111)
        losses = np.array(self.losses)
        losses_scaled = 100 * losses / losses.max()
        ax.plot(losses_scaled, label="loss")
        if self.val_acc:
            ax.plot(self.val_acc, label="validation")

        ax.legend()
        plt.show()

    def show_confusion_matrix(self, loader: DataLoader, device: torch.device) -> None:
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
        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True)
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.show()
