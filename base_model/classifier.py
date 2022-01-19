import torch
from torch import nn

class MultiSourceClassifier(nn.Module):
    def __init__(self,num_classes,in_channels,middle_channels):
        super(MultiSourceClassifier,self).__init__()
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.middle_channels=middle_channels
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.in_channels, out_features=self.middle_channels),
            nn.BatchNorm1d(self.middle_channels),
            nn.ReLU(),
            nn.Linear(in_features=self.middle_channels, out_features=self.num_classes),
        )
        self.__init__weights()

    def __init__weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self,input):
        x=self.fc(input)
        return x

class SimpleClassifier(nn.Module):
    def __init__(self,num_classes,in_channels):
        super(SimpleClassifier,self).__init__()
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.fc = nn.Linear(in_features=self.in_channels, out_features=self.num_classes)

    def forward(self,input):
        x=self.fc(input)
        return x