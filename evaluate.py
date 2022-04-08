import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import yaml
from utils import setup_logger
from dataloaders import MyDataset, split_ds
from models.Atta import AttaNet
from models.loss import Tverskyloss
import numpy as np
from medpy import metric
import pandas as pd
import matplotlib.pyplot as plt


def main(batch_size, learning_rate, seed, n_classes):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # log
    output_dir = os.path.join('eval', args.name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(output_dir)

    # data
    _, test_dict = split_ds(args.data_dir, 0.8, seed)
    test_ds = MyDataset(test_dict)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    logger.info('test_data: {}'.format(len(test_ds)))

    # model
    model = AttaNet(n_classes=n_classes)
    model.load_state_dict(torch.load(args.resume, map_location=device))
    logger.info('successful load weights: {}'.format(args.resume))
    model.to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.loss_func = Tverskyloss()

    # evaluate
    model.eval()
    evaluator = pd.DataFrame(columns=["image", "dice", "precision", "recall", "tnr"])

    for step, (images, labels) in enumerate(test_loader):
        if device == torch.device('cuda'):
            images = images.cuda()
            labels = labels.cuda()
        result = model(images)[0]
        for index in range(result.shape[0]):
            img_pth, _ = test_ds.get_path(step * batch_size + index)
            img_pth = os.path.basename(img_pth)
            image = torch.argmax(result[index], dim=0).cpu().detach().numpy()
            label = labels[index].cpu().detach().numpy()
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title('output')
            plt.xticks([]), plt.yticks([])
            plt.imshow(image, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.xticks([]), plt.yticks([])
            plt.title('label')
            plt.imshow(label, cmap='gray')
            plt.savefig(os.path.join(output_dir, img_pth))
            plt.cla()
            plt.close("all")

            # Dice系数是一种集合相似度度量函数，通常用于计算两个样本的相似度，取值范围在[0,1]
            dice = metric.dc(image.astype(np.uint8), label.astype(np.uint8))
            # 预测正确的个数占总的正类预测个数的比例（从预测结果角度看，有多少预测是准确的）
            precision = metric.precision(image.astype(np.uint8), label.astype(np.uint8))
            # 确定了正类被预测为正类占所有标注的个数（从标注角度看，有多少被召回）
            recall = metric.recall(image.astype(np.uint8), label.astype(np.uint8))
            # 真负类率(True Negative Rate),所有真实负类中，模型预测正确负类的比例
            tnr = metric.true_negative_rate(image.astype(np.uint8), label.astype(np.uint8))
            info = (img_pth, dice, precision, recall, tnr)
            evaluator.loc[step * batch_size + index] = info
            logger.info('[{}] dice: {:.3f}, precision: {:.3f}, recall: {:.3f}, tnr: {:.3f}'.
                        format(img_pth, dice, precision, recall, tnr))

    evaluator.to_csv(os.path.join(output_dir, 'evaluator.csv'), index=False)
    mean_info = evaluator[1:].mean(axis=0)
    for info in mean_info:
        logger.info(info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pupil evaluate')
    parser.add_argument('--base', default='outputs/thyroid_raw/2022-04-08-15-44-22/')
    parser.add_argument('-r', '--resume', default='epoch-100.pth')
    parser.add_argument('-c', '--config', default='eval.yaml')
    parser.add_argument('--data_dir', default='../../Datasets/thyroid_raw')
    parser.add_argument('--name', default='thyroid_raw')
    args = parser.parse_args()

    args.resume = args.base + args.resume
    args.config = args.base + args.config
    config = open(args.config)
    config = yaml.load(config, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(**config)
