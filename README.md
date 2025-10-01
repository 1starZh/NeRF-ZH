# NeRF实现

这是Neural Radiance Fields (NeRF) 的中文简单实现版本，在阅读nerf-pytorch代码的过程中，发现其对功能的处理极尽完善，
但是由于代码中嵌套过多，使得代码的可读性较差以及修改起来有比较大的困难，再与另一篇simple-nerf的实现对比之后，我打算将两者
的实现结合起来，实现一个既拥有高可读性，便于修改，又拥有比较完善的功能的nerf代码仓库。

## 项目结构

```
nerf-zh/
├── configs/               # 配置文件目录
│   ├── fern.txt           # LLFF Fern数据集配置
│   └── lego.txt           # Blender Lego数据集配置
├── data_process/          # 数据处理模块
│   ├── blender.py         # Blender数据集加载
│   └── llff.py            # LLFF数据集加载
├── checkpoints/           # 模型 checkpoint 保存目录
├── render_result/         # 渲染结果保存目录
├── log/                   # 日志保存目录
├── model.py               # NeRF 模型定义
├── train.py               # 训练脚本
├── render.py              # 渲染脚本
├── tools.py               # 工具函数
├── loss.py                # 损失函数
├── get_args.py            # 命令行参数解析
└── requirements.txt       # 依赖项列表
```

## 环境配置

### 安装依赖项

```bash
pip install -r requirements.txt
```

主要依赖项包括：
- torch: PyTorch深度学习框架
- numpy: 数值计算库
- imageio: 图像处理库
- opencv-python: 计算机视觉库
- tqdm: 进度条显示
- configargparse: 配置文件解析
- tensorboard: 训练可视化

## 使用方法

### 1. 准备数据集

项目支持两种数据集格式：
- **LLFF (Local Light Field Fusion)**：真实场景数据集
- **Blender**：合成场景数据集

请将数据集放置在 `./data/` 目录下，或通过配置文件中的 `datadir` 参数指定路径。

### 2. 训练模型

使用配置文件运行训练脚本：

```bash
python train.py --config configs/fern.txt
```

或使用Blender数据集：

```bash
python train.py --config configs/lego.txt
```

### 3. 仅渲染视频

如果只想使用已训练的模型渲染视频：

```bash
python train.py --config configs/fern.txt --render_video
```

## 配置文件说明

配置文件包含以下主要部分：

- **基本参数**：`base_dir`, `dataset_type` 等
- **日志设置**：`log`, `log_dir`, `expname` 等
- **数据设置**：`datadir`, `factor`, `half_res` 等
- **模型参数**：`netdepth`, `netwidth`, `multires` 等
- **训练参数**：`N_rand`, `N_iter`, `lrate` 等
- **保存和渲染参数**：`save_step`, `render_factor`, `i_video` 等

## 注意事项

1. 首次运行前确保已安装所有依赖项
2. 训练时间较长，建议在GPU环境下运行
3. 对于大型数据集，可能需要调整 `chunk` 参数以避免内存溢出
4. 如需加载预训练模型，请设置 `coarse_net_use_checkpoint` 和 `fine_net_use_checkpoint` 参数

## 常见问题

1. **内存不足错误**：尝试减小 `chunk` 参数值
2. **训练不稳定**：检查学习率设置，或尝试调整 `precrop_iters` 和 `precrop_frac` 参数
3. **渲染效果不佳**：增加训练迭代次数，或调整 `Nc_samples` 和 `Nf_samples` 参数

## 参考资料

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [官方NeRF实现](https://github.com/bmild/nerf)
- [nerf-pytorch实现](https://github.com/yenchenlin/nerf-pytorch)
- [simple-nerf实现](https://github.com/yenchenlin/nerf-pytorch)