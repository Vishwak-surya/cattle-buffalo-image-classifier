import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)


class RegressionHead(nn.Module):
    def __init__(self, in_features: int, num_traits: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_traits)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)
