from typing import Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
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
        optimizer: optim.Optimizer = optim.SGD,
        criterion: nn.Module = nn.CrossEntropyLoss,
        show_every_n: int = 100,
        plot_loss: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Train the nn."""
        for epoch in range(num_epochs):

            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % show_every_n == (show_every_n - 1):  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}")
                    self.losses.append(running_loss)
                    running_loss = 0.0
                    if validationloader:
                        self.val_acc.append(self.evaluate_nn(validationloader))

        if plot_loss:
            self.plot_loss(*args, **kwargs)

    def evaluate_nn(self, loader: DataLoader) -> float:
        correct = 0
        total = 0

        with torch.no_grad():
            for data in loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.forward(images.float())
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct // total

    def plot_loss(self, *args: Any, **kwargs: Any) -> None:

        plt.figure(*args, **kwargs)

        ax = plt.subplot(111)
        ax.plot(self.losses, label="loss")
        if self.val_acc:
            ax.plot(self.val_acc, label="validation")

        ax.legend()
        plt.show()
