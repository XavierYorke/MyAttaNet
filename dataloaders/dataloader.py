import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

train_transforms = transforms.Compose([

])
val_transforms = transforms.Compose([

])


# 定义读取文件的格式
def default_loader(path):
    # return Image.open(path).resize((500, 330)).convert('RGB')
    return Image.open(path).crop((210, 200, 890, 820)).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, data_path, trans=None, mode='train'):
        super().__init__()
        self.data_path = data_path
        self.transform = trans
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.mode = mode

    def __getitem__(self, index):
        image_path = self.data_path[index]['image']
        image = default_loader(image_path)
        if self.transform is not None:
            # 数据标签转换为Tensor
            image = self.transform(image)
        image = self.to_tensor(image)
        if self.mode == 'train':
            label_path = self.data_path[index]['label']
            label = Image.open(label_path).crop((210, 200, 890, 820))
            label = np.array(label).astype(np.int64)
            label[label == 255] = 1
            return image, label
        return image

    def __len__(self):
        return len(self.data_path)

    def get_path(self, index):
        img, label = self.data_path[index]['image'], self.data_path[index]['label']
        return img, label
