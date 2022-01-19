import torch
from base_model import *
from .Multimodal import *

class En_De_Fusion(MultiModalModel):
    def __init__(self, hsi_channel, lidar_channel, num_classes):
        super().__init__()
        self.En1 = Ex_Net(hsi_channel)
        self.En2 = Ex_Net(lidar_channel)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(256,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128,64,1,1,0),
            nn.BatchNorm2d(64),
            # nn.AvgPool2d(kernel_size=2,stride=2,padding=1),
            nn.ReLU()
        )

        self.De1 = Re_Net(128, hsi_channel)
        self.De2 = Re_Net(128, lidar_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = SimpleClassifier(in_channels=64, num_classes=num_classes)
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()
        self.lamda = 1

    def forward(self,hsi,lidar):
        x1 = self.En1(hsi)
        x2 = self.En2(lidar)
        x = torch.cat([x1,x2],dim=1)
        x = self.conv1(x)

        hsi_ = self.De1(x)
        lidar_ = self.De2(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return (x, hsi_, lidar_)

    def compute_loss(self,out,hsi,lidar,labels):
        loss1 = self.criterion1(out[0], labels)
        loss2 = self.criterion2(hsi, out[1]) + self.criterion2(lidar, out[2])
        loss = loss1 + self.lamda*loss2
        return loss