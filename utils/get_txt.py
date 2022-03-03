import os
import random
import yaml


def get_txt(DATASET_DIR, TXT_DIR):
    # imgs = os.listdir(DATASET_DIR + '\image\distance')
    imgs = os.listdir(DATASET_DIR + '/image/distance')
    random.shuffle(imgs)
    train_list = imgs[:int(0.8 * len(imgs))]
    test_list = imgs[(int(0.8 * len(imgs))):]
    with open(TXT_DIR + r'/train.txt', 'w') as f:
        for img in train_list:
            # f.write(os.path.join(DATASET_DIR, 'image\distance', img))
            f.write(os.path.join(DATASET_DIR, 'image/distance', img))
            f.write(' ')
            # f.write(os.path.join(DATASET_DIR, 'mask\distance_mask', img[:-4] + 'png'))
            f.write(os.path.join(DATASET_DIR, 'mask/distance_mask', img[:-4] + 'png'))
            f.write('\n')

    with open(TXT_DIR + r'/test.txt', 'w') as f:
        for img in test_list:
            # f.write(os.path.join(DATASET_DIR, 'image\distance', img))
            f.write(os.path.join(DATASET_DIR, 'image/distance', img))
            f.write(' ')
            # f.write(os.path.join(DATASET_DIR, 'mask\distance_mask', img[:-4] + 'png'))
            f.write(os.path.join(DATASET_DIR, 'mask/distance_mask', img[:-4] + 'png'))
            f.write('\n')


if __name__ == '__main__':
    config = open('../config.yaml')
    config = yaml.load(config, Loader=yaml.FullLoader)
    data_config = config['data_config']
    # get_txt(**data_config)
