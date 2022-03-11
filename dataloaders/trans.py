from torchvision import transforms

# 图像的初始化操作
train_transforms = transforms.Compose([
    transforms.CenterCrop((1800, 2800)),
    # transforms.Resize((360, 424)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.CenterCrop((1800, 2800)),
    # transforms.Resize((360, 424)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])
