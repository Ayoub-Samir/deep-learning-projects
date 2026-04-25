from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        unfreeze_layer4: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        if pretrained:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            if unfreeze_layer4:
                for parameter in self.model.layer4.parameters():
                    parameter.requires_grad = True

            for parameter in self.model.fc.parameters():
                parameter.requires_grad = True

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(inputs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

