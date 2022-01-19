from abc import ABCMeta, abstractmethod
from torch import nn
from torch.nn.functional import softmax

class UniModalModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(UniModalModel,self).__init__()

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def compute_loss(self, out, label):
        pass

    def compute_pre_label(self, out):
        outputs = softmax(out, dim=1)  # 按行计算，dim=1
        _, pre_label = outputs.max(dim=1)  # outputs.shape:torch.Size([batch_size])
        return pre_label

    def compute_pre_score(self, out):
        outputs = softmax(out, dim=1)  # 按行计算，dim=1
        pre_score, _ = outputs.max(dim=1)  # outputs.shape:torch.Size([batch_size])
        return pre_score