## AttaNet模型的PyTorch版本复现
### 配置文件
config.yaml: 参数设置
### 数据集的准备
使用utils/get_txt.py生成对应数据集列表
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