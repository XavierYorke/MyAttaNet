import argparse
import torch
from dataloaders.dataloader import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from logger import setup_logger
from models.Atta import AttaNet
from models.loss import Tverskyloss
from sklearn.metrics import roc_auc_score
from train import train
import yaml
import time
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('-c', '--config', default=None)
    return parser.parse_args()


def main(epochs, batch_size, learning_rate, output_dir, iterations, seed):
    output_dir = os.path.join(output_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(output_dir)

    # 图像的初始化操作
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((227, 227)),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.RandomResizedCrop((227, 227)),
        transforms.ToTensor(),
    ])

    # 数据集加载
    train_data = MyDataset(txt='data/train.txt', transform=transforms.ToTensor())
    test_data = MyDataset(txt='data/test.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = AttaNet(n_classes=net_config['n_classes'])
    if not args.resume is None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        logger.info('successful load weights: {}'.format(args.resume))
    model.to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.loss_func = Tverskyloss()
    # model.metric_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
    # model.metric_name = "auc"
    dfhistory = train(model, epochs, train_loader, test_loader, iterations, logger, device)


if __name__ == '__main__':
    config = open('config.yaml')
    config = yaml.load(config, Loader=yaml.FullLoader)
    train_config = config['train_config']
    data_config = config['data_config']
    net_config = config['net_config']
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(**train_config)
