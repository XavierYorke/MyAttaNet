import onnx
import onnxruntime
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models.Atta import AttaNet
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_np(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_imgs(img_path, batch_size):
    img = Image.open(img_path).convert('L')
    trans = transforms.Compose([
        # transforms.CenterCrop((360, 424)),
        transforms.CenterCrop((1800, 2800)),
        transforms.ToTensor()]
    )
    img = trans(img)
    img = img.unsqueeze(dim=0)
    img = np.repeat(img, batch_size, axis=0)
    return img


def get_net(model_path):
    net = AttaNet(n_classes=2)
    net.eval()
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    return net


def cal_torch(net, imgs, epochs):
    torch_time = time.perf_counter()
    for i in range(epochs):
        torch_out = net(imgs)[0]
    torch_time = time.perf_counter() - torch_time
    print(torch_time)


def cal_onnx(imgs, epochs, onnx_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    # ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    ort_session = onnxruntime.InferenceSession(onnx_path,
                                               providers=['CPUExecutionProvider'])

    onnx_time = time.perf_counter()
    ort_inputs = {ort_session.get_inputs()[0].name: to_np(imgs)}
    for i in range(epochs):
        ort_outs = ort_session.run(None, ort_inputs)[0]
    onnx_time = time.perf_counter() - onnx_time
    print(onnx_time)    #32.759769648022484
# 比较 ONNX 运行时和 PyTorch 结果
# np.testing.assert_allclose(to_np(torch_out)[0], ort_outs[0], rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    # model_path = 'outputs/Eyeball/2022-03-07-17-39-34/epoch-100.pth'
    model_path = 'outputs/zy/2022-03-10-13-29-16/epoch-100.pth'

    # onnx_path = 'exports/AttaNet_360x424_1.onnx'
    onnx_path = 'exports/AttaNet_1800x2800.onnx'
    # img_path = '../../Datasets/Eye/H_G/image/distance/S4000L02_00003.JPEG'
    img_path = '../../Datasets/ZY_0310/data/image/001.png'
    imgs = get_imgs(img_path, 1)
    # torch_model = get_net(model_path)
    epochs = 1
    # cal_torch(torch_model, imgs, epochs)
    cal_onnx(imgs, epochs, onnx_path)