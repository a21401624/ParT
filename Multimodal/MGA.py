import torch
from torch import nn
from .Multimodal import *

class Spectral_Net(nn.Module):
    def __init__(self, hsi_channel):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(hsi_channel,256),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU()
        )

    def forward(self,x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x = torch.cat([x1,x2,x3],dim=1)
        x = self.block4(x)
        return x


class MGANet(nn.Module):
    def __init__(self, x1_channels, x2_channels):
        super().__init__()
        self.HSI_block1 = nn.Sequential(
            nn.Conv2d(x1_channels,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.LiDAR_block1 = nn.Sequential(
            nn.Conv2d(x2_channels,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.HSI_block2 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,1,1),
            nn.Sigmoid()
        )
        self.HSI_block3 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,1,1),
            nn.Sigmoid()
        )
        self.LiDAR_block2 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,1,1),
            nn.Sigmoid()
        )
        self.LiDAR_block3 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,1,1),
            nn.Sigmoid()
        )
        self.HSI_block4 = nn.Sequential(
            nn.Conv2d(192,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1)
        )
        self.LiDAR_block4 = nn.Sequential(
            nn.Conv2d(192,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1)
        )

    def forward(self,hsi,lidar):
        hsi1 = self.HSI_block1(hsi)
        lidar1 = self.LiDAR_block1(lidar)

        hsi_att1 = self.HSI_block2(hsi1)
        lidar_att1 = self.LiDAR_block2(lidar1)
        hsi2 = hsi1 + hsi1 * lidar_att1
        lidar2 = lidar1 + lidar1 * hsi_att1

        hsi_att2 = self.HSI_block3(hsi2)
        lidar_att2 = self.LiDAR_block3(lidar2)
        hsi3 = hsi2 + hsi2 * lidar_att2
        lidar3 = lidar2 + lidar2 * hsi_att2

        hsi4 = torch.cat([hsi1,hsi2,hsi3],dim=1)
        lidar4 = torch.cat([lidar1,lidar2,lidar3],dim=1)

        hsi = self.HSI_block4(hsi4)
        lidar = self.LiDAR_block4(lidar4)
        return hsi, lidar


class MGA(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes):
        super().__init__()
        self.spectral_hsi = Spectral_Net(hsi_channel)
        self.MGANet = MGANet(hsi_channel, lidar_channel)
        
        self.classifier = nn.Linear(640, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar):
        x1, x2 = self.MGANet(hsi, lidar)
        x3 = self.spectral_hsi(hsi[:,:,5,5])

        x = torch.cat([x1,x2,x3],dim=1)
        x = self.classifier(x)
        return x

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss

# if __name__=='__main__':
#     hsi = torch.rand(4,144,11,11)
#     lidar = torch.rand(4,1,11,11)
#     model = MGA(144,1,15)
#     out = model(hsi,lidar)
#     print(out.shape)