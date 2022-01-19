from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch import optim

from utils import AverageMeter, SingleSourceDataset, compute_metrics, MaxIterSampler

import numpy as np
import pandas as pd
import random
import argparse
import copy
import os
import time
import yaml
from easydict import EasyDict
from shutil import copyfile

from log_test import Logger
from Unimodal import model_entry
from optim_lrscheduler import optim_entry
from custom_transforms import create_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(configs):
    transform = create_transform(configs.transforms)
    # dataset = UnimodalDataset(configs.csv_dir, configs.data_dir, 2*configs.r+1, transform)
    dataset = SingleSourceDataset(configs.csv_dir, configs.pic_dir, configs.r, transform)
    sample_num = dataset.__len__()
    if "total_iter" in configs:
        itersampler = MaxIterSampler(dataset, configs.total_iter, configs.batch_size)
        dataloader = DataLoader(dataset, batch_size=configs.batch_size, sampler=itersampler, num_workers=configs.num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers, pin_memory=True)
    return dataloader, sample_num


def exp_lr_scheduler(optimizer, current_iter, lr_decay_iter):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_iter iters."""
    decay_rate = 0.9 ** (current_iter // lr_decay_iter)
    if current_iter % lr_decay_iter == 0:
        log.info('decay_rate is set to {}'.format(decay_rate))
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
    return optimizer


def data_preprocessing(data):
    input = data['DATA'].float().cuda()
    batch_size = input.size()[0]
    labels = data['label'].view(batch_size, -1)
    labels = torch.squeeze(labels, dim=1)  # labels.shape:torch.Size([batch_size])
    labels = labels.long().cuda()
    y = data['y']
    x = data['x']
    return input, labels, batch_size, y, x

def pre_train(log_iter):
    meters = EasyDict()
    meters.losses = AverageMeter(log_iter)
    meters.acc = AverageMeter(log_iter)
    return meters

def train_model(log, log_dir, configs):
    configs = EasyDict(yaml.load(open(configs), Loader=yaml.FullLoader))

    setup_seed(0)

    model, warnings = model_entry(configs.model)
    model.cuda()
    model.train()
    log.info(model)
    log.info(warnings)

    meters = pre_train(configs.log_iter)
    tb_logger = SummaryWriter(log_dir)

    optimizer = optim_entry(model.parameters(), configs.optimizer)

    dataloader_train, train_sample_num = load_data(configs.data.train)
    dataloader_valid, val_sample_num = load_data(configs.data.valid)

    current_iter = 0
    for i, data in enumerate(dataloader_train):
        current_iter += 1
        optimizer = exp_lr_scheduler(optimizer, current_iter, configs.lr_decay_iter)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        input, labels, batch_size, _, _ = data_preprocessing(data)
        optimizer.zero_grad()
        out = model.forward(input)
        loss = model.compute_loss(out, labels)
        loss.backward()
        # for name, parms in model.named_parameters():	
        #     if name in ['proj.weight']:
        #         print('-->name:', name)
        #         print('-->para:', parms)
        #         print('-->grad_requirs:',parms.requires_grad)
        #         print('-->grad_value:',parms.grad)
        #         print("===")
        optimizer.step()

        Pre_label = model.compute_pre_label(out)
        pre_true_num = torch.sum(torch.eq(Pre_label, labels)).item()
        acc = pre_true_num / batch_size

        meters.losses.update(loss.item())
        meters.acc.update(acc)

        if current_iter % configs.log_iter == 0:
            tb_logger.add_scalar('loss/loss_train', meters.losses.avg, current_iter)
            tb_logger.add_scalar('acc/acc_train', meters.acc.avg, current_iter)
            tb_logger.add_scalar('lr', current_lr, current_iter)
            log.info("train iter %d, lr=%f, loss=%f(%f), acc=%f(%f)"
                        % (current_iter, current_lr, meters.losses.val, meters.losses.avg, meters.acc.val, meters.acc.avg))

        if current_iter != 0 and current_iter % configs.valid_iter == 0:
            model.eval()

            Labels = []
            Predict_labels = []
            epoch_loss = 0
            total_pre_true = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(dataloader_valid):
                    input, labels, batch_size, y, x = data_preprocessing(data)
                    out = model.forward(input)
                    loss = model.compute_loss(out, labels)
                    epoch_loss += loss.cpu().item() * batch_size

                    Pre_label = model.compute_pre_label(out)
                    pre_true_num = torch.sum(torch.eq(Pre_label, labels)).item()
                    total_pre_true += pre_true_num

                    Labels.extend(labels.detach().cpu().numpy().flatten().tolist())
                    Predict_labels.extend(Pre_label.detach().cpu().numpy().flatten().tolist())

                tb_logger.add_scalar('loss/loss_valid', epoch_loss / val_sample_num, current_iter)
                tb_logger.add_scalar('acc/acc_valid', total_pre_true / val_sample_num, current_iter)
                log.info("valid iter %d, valid loss=%f, valid acc=%f"
                        % (current_iter, epoch_loss / val_sample_num, total_pre_true / val_sample_num))

            model.train()
    
        if current_iter != 0 and current_iter % configs.save_iter == 0:
            if not os.path.exists(os.path.join(log_dir, "checkpoints")):
                os.makedirs(os.path.join(log_dir, "checkpoints"))
            best_model = copy.deepcopy(model)
            torch.save(best_model.cpu().state_dict(), os.path.join(log_dir, "checkpoints", str(current_iter) + ".pth"))
            log.info('save checkpoint, iter %d' % current_iter)

    if not os.path.exists(os.path.join(log_dir, "checkpoints")):
        os.makedirs(os.path.join(log_dir, "checkpoints"))
    torch.save(model.cpu().state_dict(), os.path.join(log_dir, "checkpoints", str(current_iter) + ".pth"))
    log.info('save checkpoint, iter %d' % current_iter)


def test_model(log, log_dir, configs):
    configs = EasyDict(yaml.load(open(configs), Loader=yaml.FullLoader))

    model, warnings = model_entry(configs.model)
    model.cuda()
    model.eval()
    log.info(warnings)

    dataloader_test, test_sample_num = load_data(configs.data.test)

    test_loss = 0
    Labels = []
    Predict_labels = []
    Predict_scores = []
    Y = []
    X = []
    for batch_idx, data in enumerate(dataloader_test):
        input, labels, batch_size, y, x = data_preprocessing(data)
        out = model.forward(input)
        loss = model.compute_loss(out, labels)
        test_loss += loss.cpu().item() * batch_size

        Pre_label = model.compute_pre_label(out)
        Pre_score = model.compute_pre_score(out)

        Labels.extend(labels.detach().cpu().numpy().flatten().tolist())
        Predict_labels.extend(Pre_label.detach().cpu().numpy().flatten().tolist())
        Predict_scores.extend(Pre_score.detach().cpu().numpy().flatten().tolist())
        Y.extend(y.detach().cpu().numpy().flatten().tolist())
        X.extend(x.detach().cpu().numpy().flatten().tolist())

    oa, aa, kappa, acc_per_class = compute_metrics(Predict_labels, Labels)
    log.info("test_sample_num=%d, test_loss=%f, test acc=%f"
          % (test_sample_num, test_loss / test_sample_num, oa))
    log.info("oa=%f ,aa=%f, Kappa=%f"% (oa, aa, kappa))
    for i in range(acc_per_class.shape[1]):
        log.info("class %d, acc=%f" % (i+1, 100*acc_per_class[0,i]))
    
    results1 = pd.DataFrame(columns=['height','width','label','pred_label'])
    results2 = pd.DataFrame(columns=['pred_score'])
    for i in range(len(Labels)):
        results1.loc[i]=[Y[i],X[i],Labels[i],Predict_labels[i]]
        results2.loc[i]=[Predict_scores[i]]
    results = pd.concat([results1, results2], axis=1)
    results.to_csv(os.path.join(log_dir, 'results.csv'),sep=',',index=False)
    return oa, aa, kappa, acc_per_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('state', type=str, help='train or test')
    parser.add_argument('configs', type=str, default=None, help='location of the config yaml file')
    parser.add_argument('dir_name', type=str, default=None, help='name of subdir under logs')
    parser.add_argument('job_name', type=str, default=None, help='name of this job')
    args = parser.parse_args()
    state = args.state
    configs = args.configs
    dir_name = args.dir_name
    job_name = args.job_name

    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    time_prefix = time.strftime('%Y%m%d_%H%M%S', timestruct)
    log_dir = "logs/" + dir_name + "/" + state + "_"+ time_prefix + "_"+ job_name
    log = Logger(name=__name__, base_dir=log_dir).get_log

    copyfile(configs, os.path.join(log_dir,"backup.yaml"))
    
    if state == 'train':
        train_model(log, log_dir, configs)
    else:
        oa, aa, kappa, acc_per_class = test_model(log, log_dir, configs)