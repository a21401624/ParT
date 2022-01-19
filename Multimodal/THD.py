import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
from .Multimodal import *

def attention(q, k, d_k):
    # q.shape:[bs,N,d_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    return scores


class THDStack(nn.Module):
    def __init__(self, x1_channels, x2_channels, d_model):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(x1_channels,d_model,kernel_size=3,stride=1,padding=1)
        self.ln1 = nn.LayerNorm(d_model)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(x2_channels,d_model,kernel_size=3,stride=1,padding=1)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)

    def forward(self,x1,x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1 = rearrange(x1,'b c h w -> b (h w) c')
        x2 = rearrange(x2,'b c h w -> b (h w) c')
        x1 = self.relu1(self.ln1(x1))
        x2 = self.relu2(self.ln2(x2))

        att1 = attention(x1, x1, self.d_model)
        att2 = attention(x2, x2, self.d_model)
        x1_ = torch.matmul(att2, x1)
        x2_ = torch.matmul(att1, x2)
        x1 = self.ln3(x1 + x1_)
        x2 = self.ln4(x2 + x2_)
        return x1, x2


class THD(MultiModalModel):
    def __init__(self, hsi_channel, lidar_channel, d_model, dropout, num_classes):
        super().__init__()
        self.stack1 = THDStack(hsi_channel, lidar_channel, d_model)
        self.stack2 = THDStack(d_model, d_model, d_model)
        self.stack3 = THDStack(d_model, d_model, d_model)
        self.stack4 = THDStack(d_model, d_model, d_model)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(dropout),
            nn.Flatten(1),
            nn.Linear(d_model*8 + hsi_channel + lidar_channel, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar):
        x1,x2 = self.stack1(hsi, lidar)
        x1 = rearrange(x1,'b (h w) c -> b c h w', h=11, w=11)
        x2 = rearrange(x2,'b (h w) c -> b c h w', h=11, w=11)

        x3,x4 = self.stack2(x1, x2)
        x3 = rearrange(x3,'b (h w) c -> b c h w', h=11, w=11)
        x4 = rearrange(x4,'b (h w) c -> b c h w', h=11, w=11)

        x5,x6 = self.stack3(x3, x4)
        x5 = rearrange(x5,'b (h w) c -> b c h w', h=11, w=11)
        x6 = rearrange(x6,'b (h w) c -> b c h w', h=11, w=11)

        x7,x8 = self.stack4(x5, x6)
        x7 = rearrange(x7,'b (h w) c -> b c h w', h=11, w=11)
        x8 = rearrange(x8,'b (h w) c -> b c h w', h=11, w=11)

        x = torch.cat([hsi,lidar,x1,x2,x3,x4,x5,x6,x7,x8],dim=1)
        x = self.classifier(x)
        return x

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss