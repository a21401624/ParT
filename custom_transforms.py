import numpy as np
import torch
from torchvision import transforms

class RandomNoise(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if len(sample) == 4:
            data, label, y, x= sample['DATA'], sample['label'], sample['y'], sample['x']
            if np.random.random() < self.p:
                noise = np.float32(np.random.normal(0.0, 0.01, size=data.shape))
                data = data + noise
            return {'DATA': data,  'label': label, 'y': y, 'x': x}
        else:
            hsi, lidar, label, y, x = sample['hsi'], sample['lidar'], sample['label'], sample['y'], sample['x']
            if np.random.random() < self.p:
                noise = np.float32(np.random.normal(0.0, 0.01, size=hsi.shape))
                hsi = hsi + noise
                noise = np.float32(np.random.normal(0.0, 0.01, size=lidar.shape))
                lidar = lidar + noise
            return {'hsi': hsi, 'lidar': lidar, 'label': label, 'y': y, 'x': x}

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if len(sample) == 4:
            data, label, y, x= sample['DATA'], sample['label'], sample['y'], sample['x']
            if np.random.random() < self.p:
                data=np.flip(data,1)
            return {'DATA': data,  'label': label, 'y': y, 'x': x}
        else:
            hsi, lidar, label, y, x = sample['hsi'], sample['lidar'], sample['label'], sample['y'], sample['x']
            if np.random.random() < self.p:
                hsi = np.flip(hsi, 1)
                lidar=np.flip(lidar,1)
            return {'hsi': hsi, 'lidar': lidar, 'label': label, 'y': y, 'x': x}

class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if len(sample) == 4:
            data, label, y, x= sample['DATA'], sample['label'], sample['y'], sample['x']
            if np.random.random() < self.p:
                data=np.flip(data,0)
            return {'DATA': data,  'label': label, 'y': y, 'x': x}
        else:
            hsi, lidar, label, y, x = sample['hsi'], sample['lidar'], sample['label'], sample['y'], sample['x']
            if np.random.random() < self.p:
                hsi = np.flip(hsi, 0)
                lidar=np.flip(lidar,0)
            return {'hsi': hsi, 'lidar': lidar, 'label': label, 'y': y, 'x': x}

class RandomRotate(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if len(sample) == 4:
            data, label, y, x= sample['DATA'], sample['label'], sample['y'], sample['x']
            if np.random.random() < self.p:
                k = np.random.randint(4)
                data = np.rot90(data, k)
            return {'DATA': data,  'label': label, 'y': y, 'x': x}
        else:
            hsi, lidar, label, y, x = sample['hsi'], sample['lidar'], sample['label'], sample['y'], sample['x']
            if np.random.random() < self.p:
                k = np.random.randint(4)
                hsi = np.rot90(hsi, k)
                lidar = np.rot90(lidar,k)
            return {'hsi': hsi, 'lidar': lidar, 'label': label, 'y': y, 'x': x}

class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) == 4:
            data, label, y, x= sample['DATA'], sample['label'], sample['y'], sample['x']
            data = torch.from_numpy(np.transpose(data, (2, 0, 1)).copy())
            label = torch.from_numpy(np.expand_dims(label, 0).copy())
            return {'DATA': data,  'label': label, 'y': y, 'x': x}
        else:
            hsi, lidar, label, y, x = sample['hsi'], sample['lidar'], sample['label'], sample['y'], sample['x']
            hsi = torch.from_numpy(np.transpose(hsi, (2, 0, 1)).copy())
            lidar = torch.from_numpy(np.transpose(lidar, (2, 0, 1)).copy())
            label = torch.from_numpy(np.expand_dims(label, 0).copy())
            return {'hsi': hsi, 'lidar': lidar, 'label': label, 'y': y, 'x': x}

transform_dict = {
    'RandomNoise': RandomNoise,
    'RandomVerticalFlip': RandomVerticalFlip,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomRotate': RandomRotate,
    'ToTensor': ToTensor,
}

def get_single_type_trans(item):
    name = list(item.keys())[0]
    if name == 'ToTensor':
        return ToTensor()
    else:
        config = list(item.values())[0]
        return transform_dict[name](**config)

def create_transform(configs):
    transform = []
    for item in configs:
        transform.append(get_single_type_trans(item))
    transform = transforms.Compose(transform)
    return transform