from torch.utils.data import Dataset
from PIL import Image


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')

# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：


class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = self.loader(img)  
        label = self.loader(label)
        if self.transform is not None:
            # 数据标签转换为Tensor
            img = self.transform(img)  
            label = self.transform(label)
        return img, label  

    def __len__(self):  
        return len(self.imgs)



