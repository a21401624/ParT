from torch.utils.data import Dataset,Sampler
import numpy as np
import pandas as pd
import tifffile as tiff
import math
from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import pickle

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            
            self.val = self.history[-1]
            self.avg = np.mean(self.history)

        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count
            

def compute_metrics(Predict_labels,Labels):
    kappa = cohen_kappa_score(np.array(Predict_labels).reshape(-1, 1), np.array(Labels).reshape(-1, 1))
    oa = accuracy_score(Labels, Predict_labels)
    # 计算AA
    con_mat = confusion_matrix(Labels, Predict_labels)
    acc_per_class = np.zeros((1, con_mat.shape[0]))
    for i in range(con_mat.shape[0]):
        acc_per_class[0, i] = con_mat[i, i] / np.sum(con_mat[i, :])
    aa = np.mean(acc_per_class)
    return oa, aa, kappa, acc_per_class

# def sample_wise_standardization(X, axis=0):
#     """
#         standardize all channels(axis=0) or all pixels(axis=1) of the sample to mean=0 ,std=1
#         Input:
#             sample
#         Output:
#             Normalized sample
#     """
#     newX = np.reshape(X, (-1, X.shape[2]))
#     newX = preprocessing.scale(newX,axis=axis)
#     newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
#     return newX

def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)

class MultiSourceDataset(Dataset):

    def __init__(self, csv_dir, hsi_dir, lidar_dir, r, transform):
        self.image_frame = pd.read_csv(csv_dir, sep=',')
        self.r = r

        hsi = np.asarray(tiff.imread(hsi_dir))
        hsi = np.pad(hsi, ((self.r, self.r), (self.r, self.r), (0, 0)), 'symmetric')
        hsi = sample_wise_standardization(hsi)  # mean=0,std=1
        self.hsi=hsi

        lidar = np.asarray(tiff.imread(lidar_dir))
        if len(lidar.shape) == 2:
            lidar = lidar[..., np.newaxis]
        lidar = np.pad(lidar, ((self.r, self.r), (self.r, self.r), (0, 0)), 'symmetric')
        lidar = sample_wise_standardization(lidar)  # mean=0,std=1
        self.lidar = lidar
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        height = self.image_frame.iloc[idx, 0]
        width = self.image_frame.iloc[idx, 1]
        label = self.image_frame.iloc[idx, 2] - 1  # label从0开始
        h, w = height + self.r, width + self.r
        hsi = self.hsi[h - self.r:h + self.r + 1, w - self.r:w + self.r + 1, :]
        lidar = self.lidar[h - self.r:h + self.r + 1, w - self.r:w + self.r + 1,:]
        sample = {'hsi': hsi, 'lidar': lidar, 'label': label, 'y': height , 'x': width }
        if self.transform:
            sample = self.transform(sample)
        return sample

class SingleSourceDataset(Dataset):

    def __init__(self, csv_dir, pic_dir, r, transform):
        self.image_frame = pd.read_csv(csv_dir,sep=',')
        self.r = r
        pic = np.asarray(tiff.imread(pic_dir))
        if len(pic.shape) == 2:
            pic = pic[..., np.newaxis]
        pic = np.pad(pic, ((self.r, self.r), (self.r, self.r), (0, 0)), 'symmetric')
        pic = sample_wise_standardization(pic)  # mean=0,std=1
        self.pic = pic
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self,idx):
        height = self.image_frame.iloc[idx, 0]
        width = self.image_frame.iloc[idx, 1]
        label = self.image_frame.iloc[idx, 2] - 1  #label从0开始
        h, w = height + self.r, width + self.r
        data = self.pic[h - self.r : h + self.r + 1, w - self.r : w + self.r + 1, :]
        sample = {'DATA': data, 'label': label, 'y': height , 'x': width }
        if self.transform:
            sample = self.transform(sample)
        return sample


#The following two datasets use the 7*7 patch data of DFC2013 provided by Hong Danfeng in his MDL_RS parer
class MDLRSDataset(Dataset):

    def __init__(self, csv_dir, hsi_dir,lidar_dir,transform):
        '''

        Args:
            csv_dir:
            hsi_dir: the pkl dir
            lidar_dir: the pkl dir
        '''
        self.image_frame = pd.read_csv(csv_dir, sep=',')
        self.hsi=pickle.load(open(hsi_dir,'rb'))
        self.lidar = pickle.load(open(lidar_dir,'rb'))
        self.transform=transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        label = self.image_frame.iloc[idx, 2] - 1  # label从0开始
        hsi=self.hsi[idx,:].reshape((7,7,144))
        lidar=self.lidar[idx,:].reshape((7,7,21))
        sample = {'hsi': hsi, 'lidar': lidar, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

#The following two datasets use the generated npy data of DFC2013
class MSDataset(Dataset):

    def __init__(self, csv_dir, hsi_dir, lidar_dir, patch_size, transform):
        '''

        Args:
            csv_dir:
            hsi_dir: the npy dir
            lidar_dir: the pkl dir
        '''
        self.image_frame = pd.read_csv(csv_dir, sep=',')
        self.hsi=np.load(hsi_dir)
        self.lidar = np.load(lidar_dir)
        self.patch_size=patch_size
        self.hsi_channel=self.hsi.shape[-1]
        self.lidar_channel=self.lidar.shape[-1]
        self.transform=transform

        if(len(self.image_frame)!=len(self.hsi)):#测试集不存在这个问题
            self.hsi=self.hsi[0:len(self.hsi):4]
            self.lidar=self.lidar[0:len(self.lidar):4]

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        height = self.image_frame.iloc[idx, 0]
        width = self.image_frame.iloc[idx, 1]
        label = self.image_frame.iloc[idx, 2] - 1  # label从0开始
        hsi=self.hsi[idx,:].reshape((self.patch_size,self.patch_size,self.hsi_channel))
        lidar=self.lidar[idx,:].reshape((self.patch_size,self.patch_size,self.lidar_channel))
        sample = {'hsi': hsi, 'lidar': lidar, 'label': label, 'y': height , 'x': width }
        if self.transform:
            sample = self.transform(sample)
        return sample

class UnimodalDataset(Dataset):

    def __init__(self, csv_dir, data_dir, patch_size, transform):
        '''

        Args:
            csv_dir:
            data_dir: the npy dir
        '''
        self.image_frame = pd.read_csv(csv_dir, sep=',')
        self.data = np.load(data_dir)
        self.patch_size = patch_size
        self.data_channel = self.data.shape[-1]
        self.transform = transform

        if(len(self.image_frame) != len(self.data)):#测试集不存在这个问题
            self.data = self.data[0:len(self.data):4]

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        height = self.image_frame.iloc[idx, 0]
        width = self.image_frame.iloc[idx, 1]
        label = self.image_frame.iloc[idx, 2] - 1  # label从0开始
        data = self.data[idx,:].reshape((self.patch_size,self.patch_size,self.data_channel))
        sample = {'DATA': data, 'label': label, 'y': height , 'x': width }
        if self.transform:
            sample = self.transform(sample)
        return sample


class MaxIterSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
    """

    def __init__(self, dataset, total_iter, batch_size, last_iter=-1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = 1
        self.last_iter = last_iter
        self.num_samples = self.batch_size * total_iter
        self.total_size = self.num_samples * self.world_size
        self.num_epoches = int(math.ceil(self.total_size * 1.0 / len(self.dataset)))
        self.call = 0

        print('=> len(dataset)={}, num_samples={}, total_size={}, num_epoches={}'
              .format(len(self.dataset), self.num_samples, self.total_size, self.num_epoches))

    def __iter__(self):
        if self.call == 0:
            self.call = 1

            indices = []
            for epoch_index in range(self.num_epoches):
                # deterministically shuffle based on epoch

                # Option 2: Numpy way
                np.random.seed(epoch_index)
                cur_indices = np.arange(len(self.dataset)).tolist()
                np.random.shuffle(cur_indices)
                indices += cur_indices

            # keep up to the predefined number of samples
            indices = indices[:self.total_size]
            # np.random.shuffle(indices) # Option 3: Deal with memory error

            assert len(indices) == self.total_size

            # indices = indices[self.rank::self.world_size]
            assert len(indices) == self.num_samples

            return iter(indices[(self.last_iter + 1) * self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def __len__(self):
        return self.num_samples