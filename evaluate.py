import os
import time
import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from utils import setup_logger
from dataloaders.dataloader import MyDataset
from models.Atta import AttaNet
from models.loss import Tverskyloss
import numpy as np
from medpy import metric
import pandas as pd
import matplotlib.pyplot as plt
from dataloaders.trans import test_transforms


def main(batch_size, learning_rate, seed, n_classes, txt_path):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # log
    output_dir = os.path.join('eval', time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(output_dir)

    test_data = MyDataset(txt=txt_path + '/test.txt', transform=test_transforms)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info('test_data: {}'.format(len(test_data)))

    # model
    model = AttaNet(n_classes=n_classes)
    model.load_state_dict(torch.load(os.path.join(args.resume, args.epoch), map_location=device))
    logger.info('successful load weights: {}'.format(os.path.join(args.resume, args.epoch)))
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
            img_pth, _ = test_data.get_path(step * batch_size + index)

            image = result[index].cpu().detach().numpy().transpose(1, 2, 0)
            image = np.squeeze(image, axis=-1)
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')

            label = labels[index].cpu().detach().numpy().transpose(1, 2, 0)
            label = label[:, :, 0]
            plt.subplot(1, 2, 2)
            plt.imshow(label, cmap='gray')
            plt.show()
            image[image >= 0.5] = 1
            image[image < 0.5] = 0
            label[label >= 0.5] = 1
            label[label < 0.5] = 0

            # Dice系数是一种集合相似度度量函数，通常用于计算两个样本的相似度，取值范围在[0,1]
            dice = metric.dc(image.astype(np.uint8), label.astype(np.uint8))
            # 预测正确的个数占总的正类预测个数的比例（从预测结果角度看，有多少预测是准确的）
            precision = metric.precision(image.astype(np.uint8), label.astype(np.uint8))
            # 确定了正类被预测为正类占所有标注的个数（从标注角度看，有多少被召回）
            recall = metric.recall(image.astype(np.uint8), label.astype(np.uint8))
            # 真负类率(True Negative Rate),所有真实负类中，模型预测正确负类的比例
            tnr = metric.true_negative_rate(image.astype(np.uint8), label.astype(np.uint8))
            info = (img_pth[-19:], dice, precision, recall, tnr)
            evaluator.loc[step * batch_size + index] = info
            logger.info('[{}] dice: {:.3f}, precision: {:.3f}, recall: {:.3f}, tnr: {:.3f}'.
                        format(img_pth[-19:], dice, precision, recall, tnr))

    evaluator.to_csv(os.path.join(output_dir, 'evaluator.csv'), index=False)
    mean_info = evaluator[1:].mean(axis=0)
    for info in mean_info:
        logger.info(info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZY evaluate')
    parser.add_argument('-r', '--resume', default='outputs/zy/2022-03-11-09-02-37')  # 'outputs/EyeBall/2022-03-03-09-18-04 star'
    parser.add_argument('-e', '--epoch', default='epoch-100.pth')
    args = parser.parse_args()

    config = open(args.resume + '/eval.yaml')
    config = yaml.load(config, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(**config)
