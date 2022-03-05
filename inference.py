import onnx
import onnxruntime
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models.Atta import AttaNet


def to_np(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

model_path = 'outputs/2022-03-03-09-18-04/epoch-100.pth'
onnx_path = 'exports/AttaNet.onnx'
img_path = 'H:\Datasets\Eye\H_G\image\distance\S4000L02_00003.JPEG'

img = Image.open(img_path).convert('RGB')
to_tensor = transforms.ToTensor()
img = to_tensor(img)
img = img.unsqueeze(dim=0)

net = AttaNet(n_classes=2)
net.eval()
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
net.load_state_dict(checkpoint)
torch_out = net(img)[0]

model = onnx.load(onnx_path)
onnx.checker.check_model(model)
ort_session = onnxruntime.InferenceSession(onnx_path)
ort_inputs = {ort_session.get_inputs()[0].name: to_np(img)}
ort_outs = ort_session.run(None, ort_inputs)[0]

# 比较 ONNX 运行时和 PyTorch 结果
np.testing.assert_allclose(to_np(torch_out)[0], ort_outs[0], rtol=1e-03, atol=1e-05)
