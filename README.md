# 三层神经网络实现的 CIFAR-10 图像分类器
本项目为2024-2025 《神经网络与深度学习》第一次课程作业，基于 NumPy 手工搭建三层神经网络分类器，在数据集 CIFAR-10 上进行训练以实现图像分类。不依赖 pytorch，tensorflow 等现成的支持自动微分的深度学习框架。
## 📁 项目结构

```
cifar10_3layer_nn/
├── ckp/  
│   └── best_model.npz              # 最优模型权重
├── utils/                          # 工具函数
│   └── visualize.py                # 权重可视化
├── configs/                        # 超参数配置
│   └── base_conig.yaml             # 基础参数配置
├── figure/                         # 可视化结果
│   ├── parameter_distributions.png # 参数分布
│   ├── /weight_heatmaps.png        # 权重热力图
│   └── training_curves.png         # 训练曲线
├── data/                           # 数据集
│   ├── batches.meta                # 数据集元信息
│   ├── data_batch_1                # 训练数据
│   ├── data_batch_2
│   ├── ...
│   ├── data_batch_5
│   ├── test_batch                  # 测试数据
│   └── readme.html                 # 数据集说明
├── src/ 
│   ├── data_loader.py              # 数据读取
│   ├── main.py                     # 主函数
│   ├── model.py                    # 模型结构与反向传播
│   ├── param_search.py             # 参数搜索
│   ├── train.py                    # 训练代码
│   └── test.py                     # 测试代码
├── main.py                         # 主程序入口
├── README.md                       # 项目说明
└── requirements.txt                # 依赖库
```

## 🚀 快速开始
### 1️⃣ 安装依赖

``` bash
pip install -r requirements.txt
``` 
### 2️⃣ 数据准备
已从 CIFAR-10官网 下载并解压到项目根目录下的 ./data/ 文件夹内：

data_batch_1 ~ data_batch_5  
test_batch  
batches.meta

### 3️⃣ 参数配置
默认使用 ./config/base_config.yaml中的train参数配置，可以对其进行修改。你也可以使用如下指令导入自定义的参数配置：
以下是基于YAML配置文件的README参数设置部分介绍模板：

---
🛠 **参数设置说明**

#### 训练模式 (`train`)
| 参数名称         | 类型     | 默认值   | 说明                                                                 |
|------------------|----------|----------|----------------------------------------------------------------------|
| `hidden_size`    | int      | 1024     | 神经网络隐藏层的维度                                               |
| `learning_rate`  | float    | 0.01     | 初始学习率，SGD冲量较大时建议初始学习率小于0.01以平稳训练                           |
| `dropout_rate`   | float    | 0.2      | 随机失活比率，用于防止过拟合                                        |
| `activation`     | str      | 'relu'   | 隐藏层激活函数，当前固定为ReLU                                      |
| `reg`            | float    | 0        | L2正则化系数，0表示不启用正则化                                     |
| `lr_decay`       | float    | 0.9      | 学习率衰减因子（每k个epoch衰减至原值的90%）                            |
| `momentum`       | float    | 0.9      | 优化器的动量参数（适用于SGD等优化器）                               |
| `data_augment`   | bool     | True     | 是否启用训练数据增强（如随机翻转、裁剪等）                          |

---

#### 测试模式 (`test`)
| 参数名称       | 类型     | 默认值                     | 说明                                 |
|----------------|----------|----------------------------|--------------------------------------|
| `model_path`   | str      | "./models/best_model.npz"  | 预训练模型权重文件的加载路径         |

---

#### 参数搜索模式 (`param_search`)
用于超参数自动优化，支持多候选值组合搜索：
- `hidden_size`: [1024] 隐藏层大小
- `learning_rate`: [0.01, 0.05] 学习率大小
- `reg`: [0.01, 0]  L2正则化强度
- `dropout_rate`: [0.2]  随机失活比率  
- `activation`: ['sigmoid', 'relu']  激活函数
- `其他`: 参考train.py中可选参数




### 4️⃣ 训练模型
你可以直接运行以下命令开始训练：

``` bash
python main.py --train
``` 
默认使用 ./config/base_config.yaml中的train参数配置，可以对其进行修改。你也可以使用如下指令导入自定义的参数配置：

``` bash
python main.py --train --config /yourpath.yaml
``` 


### 5️⃣ 进行测试
在测试集上评估指定模型的性能：

``` bash
python main.py --test
``` 
### 6️⃣ 参数搜索
自动遍历一系列超参数组合，记录验证准确率并保存最优模型及其参数，指令如下：

``` bash
python main.py --param_search
``` 
你也可以通过修改配置yaml文件中的param_search自定义搜索空间。

### 7️⃣ 参数可视化
通过修改 ./utils/visualize.py中的模型路径，自动可视化模型权重分布和权重热力图。
## ✅ 实验结果
最终模型在CIFAR-10测试集上达到了**65.00%**的准确率。考虑到模型结构的简洁性以及未引入更深层次的特征提取模块（如卷积层），这已经是一个相当不错的结果。


