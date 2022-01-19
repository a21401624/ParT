import torch
from .Unimodal import UniModalModel
from .UnimodalCNN import UnimodalCNN
from .UnimodalTrm import UnimodalTrm

__all__=['UniModalModel','UnimodalCNN','UnimodalTrm']

def model_entry(configs):
    model = globals()[configs['type']](**configs['kwargs'])
    if 'pth_dir' in configs:
        warnings = model.load_state_dict(torch.load(configs.pth_dir))
    else:
        warnings = "Successfully load model"
    return model, warnings