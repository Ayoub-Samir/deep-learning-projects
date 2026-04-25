from __future__ import annotations

from torch import nn


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(84, num_classes),
        )

    def forward(self, inputs):
        x = self.features(inputs)
        return self.classifier(x)
