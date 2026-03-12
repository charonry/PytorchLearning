# PyTorch 深度学习学习项目

这是一个用于学习 PyTorch 深度学习框架的个人项目，涵盖了从基础概念到完整训练流程的实践内容。

## 项目概述

本项目通过实际代码示例，系统学习 PyTorch 的核心组件和神经网络构建方法，包括数据加载、模型定义、训练流程和模型部署等完整流程。

## 项目结构

```
pytorchLearning/
├── learn/                      # 学习笔记目录
│   ├── neuralnetwork/          # 神经网络组件学习
│   │   ├── MyModule.py         # 基础神经网络模块
│   │   ├── MyConv.py           # 卷积层
│   │   ├── MyConv2d.py         # 二维卷积
│   │   ├── MyMaxPool.py        # 最大池化层
│   │   ├── MyLinear.py         # 全连接层
│   │   ├── MyRelu.py           # ReLU激活函数
│   │   ├── MySequential.py     # 顺序容器
│   │   ├── MyLoss.py           # 损失函数
│   │   ├── MyLossNetwork.py    # 损失函数网络应用
│   │   └── MyOptimizer.py      # 优化器
│   ├── module/                 # 模型管理
│   │   ├── MyPretrained.py     # 预训练模型
│   │   ├── MyModuleSave.py     # 模型保存
│   │   └── MyModuleLoad.py     # 模型加载
│   ├── MyDataSet.py            # 自定义数据集
│   ├── MyDataloader.py         # 数据加载器
│   ├── MyTransforms.py         # 数据转换
│   ├── MyDataSetTransforms.py  # 数据集转换
│   ├── MyConventTransforms.py  # 组合转换
│   └── MyTensorboard.py        # TensorBoard可视化
├── train/                      # 训练实战
│   ├── MyModule.py             # CNN模型定义
│   ├── MyTrain.py              # CPU训练脚本
│   ├── MyTrainGPU01.py         # GPU训练脚本v1
│   └── MyTrainGPU02.py         # GPU训练脚本v2
├── test/                       # 测试目录
│   └── MyTest.py               # 模型测试推理
├── dataset/                    # CIFAR-10数据集
├── logs/                       # TensorBoard日志
├── modulesave/                 # 保存的模型文件
└── resource/                   # 资源文件
    └── dataset_group/          # 自定义数据集(蚂蚁/蜜蜂)
```

## 核心功能

### 1. 数据加载与处理
- 使用 `Dataset` 和 `DataLoader` 加载数据
- 支持 CIFAR-10 标准数据集
- 自定义数据集（蚂蚁 vs 蜜蜂二分类）
- 图像预处理和数据增强（Resize、ToTensor等）

### 2. 神经网络构建
- 卷积神经网络（CNN）
- 包含卷积层、池化层、全连接层
- 使用 Sequential 容器组织网络结构

### 3. 训练流程
- 完整的训练-验证循环
- 支持 CPU 和 GPU（CUDA）训练
- 损失函数（CrossEntropyLoss）
- 优化器（SGD）
- 学习率设置

### 4. 模型管理
- 模型保存与加载
- 预训练模型使用
- 模型推理测试

### 5. 可视化
- TensorBoard 训练过程可视化
- 训练损失、测试损失、准确率曲线

## 快速开始

### 环境要求
- Python 3.x
- PyTorch
- torchvision
- PIL
- TensorBoard

### 运行训练
```bash
# CPU训练
python train/MyTrain.py

# GPU训练
python train/MyTrainGPU01.py
```

### 运行测试
```bash
python test/MyTest.py
```

### 查看训练可视化
```bash
tensorboard --logdir=logs
```

## 模型架构

项目中定义的 CNN 模型结构：

```
Input (3, 32, 32)
    ↓
Conv2d(3, 32, 5) + MaxPool2d(2)
    ↓
Conv2d(32, 32, 2) + MaxPool2d(2)
    ↓
Conv2d(32, 64, 5) + MaxPool2d(2)
    ↓
Flatten
    ↓
Linear(64*4*4, 64)
    ↓
Linear(64, 10) → Output
```

适用于 CIFAR-10 数据集（10类图像分类）。

## 学习要点

1. **基础概念**：张量操作、自动求导
2. **数据流程**：Dataset → DataLoader → 训练循环
3. **网络组件**：卷积、池化、全连接、激活函数
4. **训练技巧**：损失计算、反向传播、参数更新
5. **模型部署**：保存、加载、推理

## 备注

- 项目主要用于个人学习，代码包含详细注释
- 包含多个版本的实现，展示不同写法
- 部分代码为教学演示，可能有简化处理
