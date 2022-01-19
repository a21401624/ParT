import torch
from einops import rearrange
from .Multimodal import *
from base_model import *

# V0.0 baseline1
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        # self.lidar_proj = ConvNet(lidar_channel,d_model)
        self.lidar_proj = nn.Conv2d(lidar_channel, d_model, kernel_size=1, bias=False)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.hsi_Trm = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.lidar_Trm = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        lidar = self.lidar_proj(lidar)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.hsi_Trm(hsi)
        lidar = self.lidar_Trm(lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = (hsi + lidar)/2
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V1.0
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.ConvNet = ConvNet(lidar_channel,1)
        self.lidar_proj = nn.Conv2d(lidar_channel, d_model, kernel_size=1, bias=False)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.hsi_Trm = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.lidar_Trm = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        spatial_att = self.ConvNet(lidar)
        spatial_att = rearrange(spatial_att, 'b c h w -> b (h w) c')
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')
        hsi = hsi * spatial_att

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        # hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.hsi_Trm(hsi)
        lidar = self.lidar_Trm(lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = (hsi + lidar)/2
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V2.0 ParT

class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type,cutout_prob,cutout_num):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = ConvNet(lidar_channel,d_model)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)
        self.cutout = SequenceCutOut(cutout_prob, cutout_num)

        self.Trm1 = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.Trm2 = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)
        if self.training:
            hsi,lidar = self.cutout((hsi,lidar))

        hsi = self.Trm1(hsi)
        lidar = self.Trm2(lidar, hsi, lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = (hsi + lidar)/2
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss


# V2.0.1
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = ConvNet(lidar_channel,d_model)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.Trm1 = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.Trm2 = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.Trm1(hsi)
        lidar = self.Trm2(lidar, hsi, lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = hsi + lidar 
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V2.1
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = ConvNet(lidar_channel,d_model)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.Trm1 = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.Trm2 = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model*2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.Trm1(hsi)
        lidar = self.Trm2(lidar, hsi, lidar)

        x = torch.cat([hsi,lidar],dim=2)
        # x = (hsi + lidar)/2
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V2.2
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = ConvNet(lidar_channel,d_model)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.Trm1 = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.Trm2 = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.Trm1(hsi)
        lidar = self.Trm2(lidar, hsi, lidar)

        x = (torch.mean(hsi,dim=1) + torch.mean(lidar,dim=1))/2
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V2.3
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = nn.Conv2d(lidar_channel, d_model, kernel_size=1, bias=False)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.Trm1 = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.Trm2 = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.Trm1(hsi)
        lidar = self.Trm2(lidar, hsi, lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = (hsi + lidar)/2
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V2.4
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = ConvNet(lidar_channel,d_model)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.Trm1 = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.Trm2 = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.Trm1(hsi)
        lidar = self.Trm2(hsi, lidar, lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = (hsi + lidar)/2
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V3.0 baseline2
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = ConvNet(lidar_channel,d_model)

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.Trm1 = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.Trm2 = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.Trm1(hsi, lidar, hsi)
        lidar = self.Trm2(lidar, hsi, lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = (hsi + lidar)/2
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V4.0 add hsi branch
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = ConvNet(lidar_channel,d_model)
        self.hsi_branch = nn.Sequential(
            nn.Conv2d(hsi_channel,d_model,1,1,0),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model,d_model,1,1,0),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.Trm1 = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.Trm2 = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi0 = self.hsi_branch(hsi[:,:,5:6,5:6])

        hsi = self.hsi_proj(hsi)
        hsi = rearrange(hsi, 'b c h w -> b (h w) c')

        lidar = self.lidar_proj(lidar)
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.Trm1(hsi)
        lidar = self.Trm2(lidar, hsi, lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = (hsi + lidar)/2
        x = torch.mean(x,dim=1) + torch.squeeze(hsi0)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''

# V5.0  ParT + fusion_attention
'''
class Trm_Concat(MultiModalModel):
    def __init__(self,hsi_channel,lidar_channel,num_classes,d_model,N,heads,dropout,max_seq_len,d_ff,poem_type):
        super().__init__()
        self.hsi_proj = nn.Conv2d(hsi_channel, d_model, kernel_size=1, bias=False)
        self.lidar_proj = ConvNet(lidar_channel,d_model)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.attention = nn.Sequential(
            nn.Linear(d_model*2 ,d_model),
            nn.ReLU(),
            nn.Linear(d_model,2),
            nn.Softmax(dim=1)
        )

        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)

        self.hsi_Trm = TrmEncoder(d_model, N, heads, dropout, d_ff)
        self.lidar_Trm = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,hsi,lidar,mask=None):
        hsi = self.hsi_proj(hsi)
        lidar = self.lidar_proj(lidar)

        x = torch.cat([hsi,lidar],dim=1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        weight = self.attention(x)
        weight = weight[:,:,None]

        hsi = rearrange(hsi, 'b c h w -> b (h w) c')
        lidar = rearrange(lidar, 'b c h w -> b (h w) c')

        hsi = self.PoEm(hsi)
        lidar = self.PoEm(lidar)

        hsi = self.hsi_Trm(hsi)
        lidar = self.lidar_Trm(lidar)

        # x = torch.cat([hsi,lidar],dim=2)
        x = hsi * weight[:,:1,:] + lidar * weight[:,1:,:]
        x = torch.mean(x,dim=1)
        return self.classifier(x)

    def compute_loss(self,out,hsi,lidar,labels):
        loss = self.criterion(out, labels)
        return loss
'''