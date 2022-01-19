from abc import ABCMeta, abstractmethod
from torch import nn
from torch.nn.functional import softmax


class MultiModalModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(MultiModalModel,self).__init__()

    @abstractmethod
    def forward(self, hsi, lidar):
        pass

    @abstractmethod
    def compute_loss(self, out, hsi, lidar, label):
        pass

    def compute_pre_label(self, out):
        if isinstance(out,tuple):
            outputs = softmax(out[0], dim=1)  # 按行计算，dim=1
        else:
            outputs = softmax(out, dim=1)
        outputs = outputs.argmax(dim=1)  # outputs.shape:torch.Size([batch_size])
        return outputs

    def compute_pre_score(self, out):
        if isinstance(out,tuple):
            outputs = softmax(out[0], dim=1)  # 按行计算，dim=1
        else:
            outputs = softmax(out, dim=1)
        pre_score, _ = outputs.max(dim=1)  # outputs.shape:torch.Size([batch_size])
        return pre_score


