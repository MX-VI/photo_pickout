# Photo Pickout

基于深度学习的人像照片质量评分系统。

## 项目简介

Photo Pickout 是一个专门为**人像摄影师**设计的 AI 辅助工具。通过训练好的深度学习模型，可以自动对照片进行**十分制评分**（1-10分），帮助摄影师快速筛选出优质作品。

## 核心功能

- 🎯 **十分制评分**：对单张照片给出 1-10 分的质量评分
- 📊 **批量处理**：支持对整个文件夹的照片进行批量评分和排序
- 🧠 **迁移学习**：基于预训练模型，只需少量标注数据即可获得不错效果
- 📈 **可视化报告**：生成评分分布图表和详细分析报告

## 项目结构

```
photo_pickout/
├── data/                   # 数据集目录
│   ├── good/              # 高质量照片（训练集）
│   ├── bad/               # 低质量照片（训练集）
│   └── test/              # 测试照片
├── models/                # 保存训练好的模型
├── src/                   # 源代码
│   ├── train.py          # 训练脚本
│   ├── predict.py        # 评分推理脚本
│   ├── dataset.py        # 数据加载器
│   └── model.py          # 模型定义
├── configs/               # 配置文件
├── notebooks/             # Jupyter 笔记本（数据分析和实验）
├── outputs/               # 输出结果
│   ├── plots/            # 图表
│   └── reports/          # 分析报告
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone git@github.com:MX-VI/photo_pickout.git
cd photo_pickout

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

将你的照片按质量分类放入对应文件夹：

```
data/
├── good/          # 放你认为 7-10 分的照片
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
└── bad/           # 放你认为 1-6 分的照片
    ├── photo3.jpg
    ├── photo4.jpg
    └── ...
```

**建议**：
- 每个类别至少 100 张照片
- 照片越多越好（500+ 效果更佳）
- 照片应覆盖不同场景、光线、构图

### 3. 训练模型

```bash
python src/train.py --epochs 50 --batch_size 32
```

### 4. 照片评分

**单张照片评分：**
```bash
python src/predict.py --image path/to/photo.jpg
```

**批量评分并排序：**
```bash
python src/predict.py --folder path/to/photos/ --output outputs/results.csv --sort
```

## 评分标准参考

| 分数 | 等级 | 描述 |
|------|------|------------|
| 9-10 | 优秀 | 构图完美，光线极佳，表情自然，可作为样片 |
| 7-8  | 良好 | 整体不错，有小瑕疵，可交付客户 |
| 5-6  | 一般 | 有明显问题，需要修图或谨慎使用 |
| 1-4  | 较差 | 严重缺陷，建议删除或重拍 |

## 技术细节

- **基础模型**: EfficientNet-B3（预训练于 ImageNet）
- **输出**: 回归问题，输出 1-10 的连续分数
- **损失函数**: MSE Loss
- **优化器**: Adam with cosine annealing

## 模型改进建议

1. **增加细粒度标签**：将照片按构图、曝光、对焦、表情等维度分别打分
2. **多模型集成**：训练多个模型取平均，提高稳定性
3. ** active learning**：让模型推荐不确定的照片供人工标注
4. **迁移到手机端**：使用 ONNX 导出，部署到移动端

## 许可证

MIT License

## 作者

MX-VI - 人像摄影师
