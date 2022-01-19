import torch
from .Multimodal import MultiModalModel
from .Simple_Concat import Simple_Concat
from .En_De_Fusion import En_De_Fusion
from .Cross_Fusion import Cross_Fusion
from .Attention_Fusion import Attention_Fusion
from .Coupled_CNN import CoupledCNN
from .MSTrm import MSTrm
from .Trm_Fusion import Trm_Fusion
from .Trm_Concat import Trm_Concat
from .CNNTrm import CNNTrm
from .EMFNet import EMFNet
from .THD import THD
from .MGA import MGA

__all__=['MultiModalModel','Simple_Concat','En_De_Fusion','Cross_Fusion','CoupledCNN',
         'Trm_Fusion','MSTrm','Attention_Fusion','Trm_Concat','CNNTrm','EMFNet','THD','MGA']

def model_entry(configs):
    model = globals()[configs['type']](**configs['kwargs'])
    if 'pth_dir' in configs:
        warnings = model.load_state_dict(torch.load(configs.pth_dir))
    else:
        warnings = "Successfully build model"
    return model, warnings