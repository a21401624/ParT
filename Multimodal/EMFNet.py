import torch
from torch import nn
from .Multimodal import *

class EMFNet(MultiModalModel):
    def __init__(self,hsi_channel, lidar_channel, num_classes):
        super().__init__()
        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(hsi_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.lidar_conv1 = nn.Sequential(
            nn.Conv2d(lidar_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.lidar_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.lidar_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.feature_fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,2),
            nn.Softmax(dim=1)
        )

        self.channel_tunning1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            nn.Linear(32,2),
            nn.ReLU(),
            nn.Linear(2,32),
            nn.Sigmoid()
        )
        self.spatial_tunning1 = nn.Sequential(
            nn.Conv2d(32,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.hsi_mp1 = nn.MaxPool2d(2)
        self.lidar_mp1 = nn.MaxPool2d(2)

        self.channel_tunning2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            nn.Linear(64,2),
            nn.ReLU(),
            nn.Linear(2,64),
            nn.Sigmoid()
        )
        self.spatial_tunning2 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.hsi_mp2 = nn.MaxPool2d(2)
        self.lidar_mp2 = nn.MaxPool2d(2)

        self.channel_tunning3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            nn.Linear(128,2),
            nn.ReLU(),
            nn.Linear(2,128),
            nn.Sigmoid()
        )
        self.spatial_tunning3 = nn.Sequential(
            nn.Conv2d(128,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.hsi_mp3 = nn.MaxPool2d(2)
        self.lidar_mp3 = nn.MaxPool2d(2)

        self.classifier = nn.Linear(128, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar):
        hsi1 = self.hsi_conv1(hsi)
        lidar1 = self.lidar_conv1(lidar)

        x = torch.cat((hsi1,lidar1),dim=1)
        weight = self.feature_fusion(x).unsqueeze(2).unsqueeze(3)

        hsi2 = self.spatial_tunning1(lidar1) * hsi1
        hsi2 = self.hsi_mp1(hsi2)
        lidar2 = self.channel_tunning1(hsi1).unsqueeze(2).unsqueeze(3) * lidar1
        lidar2 = self.lidar_mp1(lidar2)

        hsi2 = self.hsi_conv2(hsi2)
        lidar2 = self.lidar_conv2(lidar2)

        hsi3 = self.spatial_tunning2(lidar2) * hsi2
        hsi3 = self.hsi_mp2(hsi3)
        lidar3 = self.channel_tunning2(hsi2).unsqueeze(2).unsqueeze(3) * lidar2
        lidar3 = self.lidar_mp2(lidar3)

        hsi3 = self.hsi_conv3(hsi3)
        lidar3 = self.lidar_conv3(lidar3)

        hsi4 = self.spatial_tunning3(lidar3) * hsi3
        hsi4 = self.hsi_mp3(hsi4)
        lidar4 = self.channel_tunning3(hsi3).unsqueeze(2).unsqueeze(3) * lidar3
        lidar4 = self.lidar_mp3(lidar4)

        x = hsi4 * weight[:,0:1,:,:] + lidar4 * weight[:,1:2,:,:]
        x = torch.flatten(x,1)
        out = self.classifier(x)
        return out

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss

if __name__=='__main__':
    hsi = torch.rand(4,144,11,11)
    lidar = torch.rand(4,1,11,11)
    model = EMFNet(144,1,15)
    out = model(hsi,lidar)