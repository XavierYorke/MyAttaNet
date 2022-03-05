import argparse
import torch
from dataloaders.dataloader import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import setup_logger
from models.Atta import AttaNet
from models.loss import Tverskyloss
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
    output_dir = os.path.join(output_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存eval信息
    with open(os.path.join(output_dir, 'eval.yaml'), 'w') as f:
        f.write('batch_size: {}\n'.format(batch_size))
        f.write('learning_rate: {}\n'.format(learning_rate))
        f.write('seed: {}\n'.format(seed))
        f.write('n_classes: {}\n'.format(net_config['n_classes']))

    # 启动log
    logger = setup_logger(output_dir)
    for data in str(config).split(', '):
        logger.info(data)

    # 图像的初始化操作
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])

    # 数据集加载
    train_data = MyDataset(txt='data/train.txt', transform=train_transforms)
    test_data = MyDataset(txt='data/test.txt', transform=test_transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info('train_data: {}, test_data: {}'.format(len(train_data), len(test_data)))

    # 模型
    model = AttaNet(n_classes=net_config['n_classes'])
    start = 1
    if not args.resume is None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        logger.info('successful load weights: {}'.format(args.resume))
        import re
        start = int(re.search('epoch-(\d+)', args.resume).group(1)) + 1
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.loss_func = Tverskyloss()
    model.to(device)
    train(model, start, epochs, train_loader, test_loader, iterations, logger, device, output_dir)


if __name__ == '__main__':
    args = parse_args()
    config = open(args.config)
    config = yaml.load(config, Loader=yaml.FullLoader)
    train_config = config['train_config']
    data_config = config['data_config']
    net_config = config['net_config']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(**train_config)
