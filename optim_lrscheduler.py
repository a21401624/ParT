from torch import optim, relu
from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam
import torch

def optim_entry(params, config):
    return globals()[config['type']](params, **config['kwargs'])