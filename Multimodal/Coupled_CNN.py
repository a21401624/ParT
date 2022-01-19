from .Multimodal import *

class CoupledCNN(MultiModalModel):
    def __init__(self, hsi_channel, lidar_channel, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = hsi_channel,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.5),
        )
        self.out1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=lidar_channel,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),  # BN can improve the accuracy by 4%-5%
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )

        self.out2 = nn.Sequential(
            # nn.Linear(FM*4, FM*2),
            nn.Linear(128, num_classes),
        )

        self.out3 = nn.Sequential(
            # nn.Linear(FM*4*2, Classes),
            # nn.ReLU(),
            nn.Linear(128,num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.lamda=0.01

    def forward(self, x1, x2):

        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = x1.view(x1.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out1 = self.out1(x1)

        x2 = self.conv4(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        x = x1 + x2
        out = self.out3(x)
        return (out, out1, out2)

    def compute_loss(self,out, hsi, lidar, label):
        loss = self.criterion(out[0],label) + self.lamda * self.criterion(out[1],label) + self.lamda * self.criterion(out[2],label)
        return loss