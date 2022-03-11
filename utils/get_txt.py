import os
import random
import yaml
from torchvision import transforms
from PIL import Image


def get_txt(dataset_dir, txt_dir):
    # imgs = os.listdir(os.path.join('../', dataset_dir, 'image/distance'))
    imgs = os.listdir(os.path.join('../', dataset_dir, 'image'))
    random.shuffle(imgs)
    train_list = imgs[:int(0.8 * len(imgs))]
    test_list = imgs[(int(0.8 * len(imgs))):]
    with open('../' + txt_dir + r'/train.txt', 'w') as f:
        for img in train_list:
            # f.write(os.path.join(dataset_dir, 'image/distance', img))
            f.write(os.path.join(dataset_dir, 'image', img))
            f.write(' ')
            # f.write(os.path.join(dataset_dir, 'mask/distance_mask', img[:-4] + 'png'))
            f.write(os.path.join(dataset_dir, 'label', img))
            f.write('\n')

    with open('../' + txt_dir + r'/test.txt', 'w') as f:
        for img in test_list:
            # f.write(os.path.join(dataset_dir, 'image/distance', img))
            f.write(os.path.join(dataset_dir, 'image', img))
            f.write(' ')
            # f.write(os.path.join(dataset_dir, 'mask/distance_mask', img[:-4] + 'png'))
            f.write(os.path.join(dataset_dir, 'label', img))
            f.write('\n')


if __name__ == '__main__':
    config = open('../config.yaml')
    config = yaml.load(config, Loader=yaml.FullLoader)
    data_config = config['data_config']
    # get_txt(**data_config)
    # path = data_config['dataset_dir']
    # imgs = os.listdir(os.path.join('../', path, 'image'))
    # center_crop = transforms.CenterCrop((1800, 2800))
    # for img in imgs:
    #     img = Image.open(os.path.join('../', path, 'image', img)).convert('L')
    #     img = center_crop(img)
    #     img.show()
    #     break