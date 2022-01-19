import torch
from .Unimodal import *
from base_model import *

class UnimodalCNN(UniModalModel):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.backbone = Ex_Net(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = SimpleClassifier(num_classes, 128)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input):
        x = self.backbone(input)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def compute_loss(self, out, labels):
        loss = self.criterion(out,labels)
        return loss