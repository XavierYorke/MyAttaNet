import argparse
import torch
from dataloaders import MyDataset, split_ds, train_transforms, val_transforms
from torch.utils.data import DataLoader
from utils import setup_logger
from models import AttaNet, Tverskyloss, OhemCELoss
from train import train
import yaml
import time
import os
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('-c', '--config', default='config.yaml')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main(epochs, batch_size, learning_rate, output_dir, iterations, seed):
    # 设置随机种子
    set_seed(seed)

    # 设置输出目录
    output_dir = os.path.join('outputs', output_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存eval信息
    with open(os.path.join(output_dir, 'eval.yaml'), 'w') as f:
        f.write('batch_size: {}\n'.format(batch_size))
        f.write('learning_rate: {}\n'.format(learning_rate))
        f.write('seed: {}\n'.format(seed))
        f.write('n_classes: {}\n'.format(net_config['n_classes']))
        # f.write('output_dir: {}\n'.format(output_dir))

    # 启动log
    logger = setup_logger(output_dir)
    for data in str(config).split(', '):
        logger.info(data)



    # 数据集加载
    train_dict, val_dict = split_ds(data_config['dataset_dir'], 0.8)
    train_ds = MyDataset(train_dict, train_transforms)
    val_ds = MyDataset(val_dict, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    logger.info('train_ds: {}, val_ds: {}'.format(len(train_ds), len(val_ds)))

    # 模型
    model = AttaNet(n_classes=net_config['n_classes'])
    start = 1
    if not args.resume is None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        logger.info('successful load weights: {}'.format(args.resume))
        import re
        start = int(re.search('epoch-(\d+)', args.resume).group(1)) + 1
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # model.loss_func = Tverskyloss()
    score_thres = 0.7
    n_min = 330 * 500 // 2
    ignore_idx = 255
    model.criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    model.criteria_aux1 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    model.criteria_aux2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    model.to(device)
    train(model, start, epochs, train_loader, val_loader, iterations, logger, device, output_dir)


if __name__ == '__main__':
    args = parse_args()
    config = open(args.config)
    config = yaml.load(config, Loader=yaml.FullLoader)
    train_config = config['train_config']
    data_config = config['data_config']
    net_config = config['net_config']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(**train_config)
