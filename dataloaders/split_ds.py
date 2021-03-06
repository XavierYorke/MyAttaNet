#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Time    :   2022/4/7
# Author  :   XavierYorke
# Contact :   mzlxavier1230@gmail.com

import os
import os.path as osp
import random


def split_ds(data_path, split, seed):
    """
    └─data_path
        ├─images
        │  ├─train
        │  └─val
        └─labels
            ├─train
            └─val
    """
    images_path = []
    labels_path = []
    image_root = osp.join(data_path, 'images')
    # image_root = osp.join(data_path, 'SegmentationClass')

    for curr_path, sec_paths, curr_files in os.walk(image_root):
        # print(curr_path)
        for sec_path in sec_paths:
            sec_path = osp.join(curr_path, sec_path)
            # print(sec_path)
            for _, _, files_name in os.walk(sec_path):
                for file_name in files_name:
                    file_path = osp.join(sec_path, file_name)

                    images_path.append(file_path)
                    labels_path.append(file_path.replace('images', 'labels').replace('jpg', 'png'))

    train_list = []
    val_list = []
    # 已经划分好训练集和验证集
    if not split:
        for i, (image, label) in enumerate(zip(images_path, labels_path)):
            if 'train' in image:
                train_list.append((image, label))
            if 'val' in image:
                val_list.append((image, label))

    # 固定随机种子随机划分训练集和验证集
    else:
        total_size = len(images_path)
        total_index = [i for i in range(total_size)]
        random.seed(seed)
        random.shuffle(total_index)

        train_index = total_index[:int(total_size * split)]
        val_index = total_index[int(total_size * split):]

        for i, (image, label) in enumerate(zip(images_path, labels_path)):
            if i in train_index:
                train_list.append((image, label))
            if i in val_index:
                val_list.append((image, label))
    train_dic = [{'image': image, 'label': label}
                 for image, label in train_list]
    val_dic = [{'image': image, 'label': label} for image, label in val_list]

    return train_dic, val_dic


def get_test(data_dir):
    images_list = []
    for curr_path, _, curr_files in os.walk(data_dir):
        for file in curr_files:
            file_path = os.path.join(curr_path, file)
            images_list.append(file_path)
    test_dict = [{'image': image} for image in images_list]
    return test_dict


def visualization(train_dict):
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    print(train_dict[0])
    image = Image.open(train_dict[0]['image']).crop((210, 200, 890, 820))
    # label = Image.open(train_dict[0]['label']).crop((210, 200, 890, 820))
    # pixels = 1260 * 910 * 3
    # label = np.array(label) * 80
    # label = np.array(label)
    # print(label.shape)
    # print(np.sum(np.where(label > 0, 1, 0)) / pixels)
    # print(label.size)   # (1260, 910)
    # print(np.unique(label))
    # plt.subplot(1, 2, 1)
    plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(label)
    plt.show()


if __name__ == '__main__':
    # path = r'../../Datasets\Eye_0325'
    path = r'E:\GLX\Datasets\thyroid_raw'
    test_dir = r'E:\GLX\Datasets\thyroid_data\test'
    # path = r'E:\GLX\Datasets\VOCdevkit\VOC2012'
    # train_dic, val_dic = split_ds(path, 0.8, 1228)
    test_dic = get_test(test_dir)
    visualization(test_dic)

