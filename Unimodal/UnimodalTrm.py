from torch import nn
from einops import rearrange
from .Unimodal import UniModalModel
from base_model import PositionalEmbedding, TrmEncoder, SimpleClassifier, ConvNet, SequenceCutOut

class UnimodalTrm(UniModalModel):
    def __init__(self, in_channels, num_classes, d_model, N, heads, dropout, max_seq_len, d_ff, poem_type, cutout_prob, cutout_num):
        super().__init__()
        # self.proj = ConvNet(in_channels, d_model)
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False)
        
        if poem_type=='none':
            self.PoEm = nn.Identity()
        else:
            self.PoEm = PositionalEmbedding(d_model, max_seq_len, poem_type)
        # self.dropout = nn.Dropout(dropout)
        self.cutout = SequenceCutOut(cutout_prob, cutout_num)

        self.Trm = TrmEncoder(d_model, N, heads, dropout, d_ff)

        self.classifier = SimpleClassifier(num_classes, d_model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input):
        # input = self.ConvNet(input)
        x = self.proj(input)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.PoEm(x)
        # x = self.dropout(x)
        if self.training:
            x = self.cutout(x)

        x = self.Trm(x)

        x = x.mean(dim=1)
        return self.classifier(x)

    def compute_loss(self, out, labels):
        loss = self.criterion(out,labels)
        return loss