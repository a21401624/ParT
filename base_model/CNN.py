from torch import nn
from torch.nn import Module

class Ex_Net(Module):
    def __init__(self,in_channels,layernum=4,avgpool=False):
        super(Ex_Net, self).__init__()
        self.in_channels=in_channels
        self.avgpool=avgpool
        assert layernum==3 or layernum==4
        self.layernum=layernum

        self.conv1 = nn.Conv2d(self.in_channels,16,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        if self.layernum==4:
            self.conv4 = nn.Conv2d(64,128, kernel_size=1, stride=1, padding=0)
            self.bn4=nn.BatchNorm2d(128)
            self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        if self.avgpool:
            self.gap=nn.AdaptiveAvgPool2d((1,1))

    def forward(self,input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.layernum==4:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.mp2(x)
            x = self.relu(x)

        if self.avgpool:
            x=self.gap(x)

        return x


class Re_Net(Module):
    def __init__(self,in_channels,out_channels):
        super(Re_Net,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.dconv1 = nn.ConvTranspose2d(self.in_channels, 64, kernel_size=3, stride=1, padding=0)
        self.dconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.dconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.dconv4 = nn.ConvTranspose2d(16, self.out_channels, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        x = self.dconv1(input)
        x = self.sigmoid(x)

        x = self.dconv2(x)
        x = self.sigmoid(x)

        x = self.dconv3(x)
        x = self.sigmoid(x)

        x = self.dconv4(x)
        x = self.sigmoid(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = self.ConvBlock(kernel_size=3)

    def ConvBlock(self, kernel_size):
        block = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        return block

    def forward(self, input):
        x = self.block1(input)
        return x