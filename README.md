## AttaNet模型的PyTorch版本复现
### 配置文件
config.yaml: 参数设置
### 数据集的准备
文件夹格式
```bash
├─data_dir
│  ├─images
│  │  └─train
│  └─labels
│      └─train
```
### 训练模型
```basic
python main.py
```
### 评价系数
```basic
python evaluate.py
```
### 导出模型为ONNX
```basic
python export.py
```
### 推理
```basic
python inference.py
```