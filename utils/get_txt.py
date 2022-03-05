import os
import random
import yaml


def get_txt(dataset_dir, txt_dir):
    imgs = os.listdir(dataset_dir + '/image/distance')
    random.shuffle(imgs)
    train_list = imgs[:int(0.8 * len(imgs))]
    test_list = imgs[(int(0.8 * len(imgs))):]
    with open(txt_dir + r'/train.txt', 'w') as f:
        for img in train_list:
            f.write(os.path.join(dataset_dir, 'image/distance', img))
            f.write(' ')
            f.write(os.path.join(dataset_dir, 'mask/distance_mask', img[:-4] + 'png'))
            f.write('\n')

    with open(txt_dir + r'/test.txt', 'w') as f:
        for img in test_list:
            f.write(os.path.join(dataset_dir, 'image/distance', img))
            f.write(' ')
            f.write(os.path.join(dataset_dir, 'mask/distance_mask', img[:-4] + 'png'))
            f.write('\n')


if __name__ == '__main__':
    config = open('../config.yaml')
    config = yaml.load(config, Loader=yaml.FullLoader)
    data_config = config['data_config']
    # get_txt(**data_config)
