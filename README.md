# Photo Pickout

基于深度学习的人像照片**多维度质量评分系统**。

## 项目简介

Photo Pickout 是专为人像摄影师设计的 AI 辅助工具。与普通的单维度评分不同，它能从**构图、表情、身体动作**三个专业维度分别评分，并给出综合总分（十分制）。

## 核心功能

- 🎯 **三维度评分**：
  - 构图评分：画面布局、主体位置、背景处理
  - 表情评分：面部情绪、眼神、自然度
  - 动作评分：身体姿态、手势、整体协调
- 📊 **综合总分**：基于三个维度的加权计算
- 📁 **批量处理**：对整个文件夹的照片进行批量评分和排序
- 🏷️ **灵活标注**：支持简单分类（good/bad）或精细多维度标注

## 项目结构

```
photo_pickout/
├── data/                   # 数据集目录
│   ├── good/              # 高质量照片
│   ├── bad/               # 低质量照片
│   └── annotations.json   # 多维度标注文件（可选）
├── models/                # 保存训练好的模型
├── src/                   # 源代码
│   ├── train.py          # 训练脚本
│   ├── predict.py        # 评分推理脚本
│   ├── dataset.py        # 数据加载器
│   ├── model.py          # 多维度模型定义
│   └── annotate.py       # 标注工具
├── configs/               # 配置文件
├── outputs/               # 输出结果
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

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

#### 方式一：简单分类（快速开始）

将照片按质量分类放入对应文件夹：

```
data/
├── good/          # 7-10 分的照片
└── bad/           # 1-6 分的照片
```

**建议数量**：每类至少 100 张，500+ 张效果更佳

#### 方式二：多维度精细标注（推荐）

使用标注工具为每张照片打上三个维度的分数：

```bash
# 命令行标注模式
python src/annotate.py ./data/good --output ./data/annotations.json

# 或图形界面模式（需要桌面环境）
python src/annotate.py ./data/good --gui
```

标注格式：
```json
{
  "photo_001.jpg": {
    "composition": 8.5,
    "expression": 9.0,
    "pose": 7.5,
    "overall": 8.3
  }
}
```

启用精细标注后，修改 `configs/config.yaml`：
```yaml
data:
  annotations_file: "./data/annotations.json"
```

### 3. 训练模型

```bash
python src/train.py --epochs 50 --batch_size 16
```

训练输出：
```
Epoch [1/50]
Train Loss: 0.2341 | Val Loss: 0.1987 | LR: 0.000100
  MAE - 构图: 0.512 | 表情: 0.487 | 动作: 0.623 | 总分: 0.398
```

### 4. 照片评分

**单张照片多维度评分：**
```bash
python src/predict.py --image path/to/photo.jpg
```

输出示例：
```
==================================================
照片: portrait_001.jpg
==================================================
构图评分: 8.50/10  [良好]
表情评分: 9.20/10  [优秀]
动作评分: 7.30/10  [良好]
--------------------------------------------------
综合总分: 8.40/10  [良好]
==================================================
```

**批量评分并生成报告：**
```bash
python src/predict.py --folder path/to/photos/ --output outputs/ --sort --top 20
```

输出文件：
- `scoring_results.csv` - 详细评分数据
- `score_distributions.png` - 各维度分数分布图
- `dimension_comparison.png` - 各维度平均分对比
- `summary.txt` - 汇总统计报告

## 评分标准参考

### 构图 (Composition)
| 分数 | 等级 | 描述 |
|------|------|------|
| 9-10 | 优秀 | 画面平衡、主体突出、背景简洁、线条引导出色 |
| 7-8  | 良好 | 构图合理，有小瑕疵但不影响整体 |
| 5-6  | 一般 | 构图有明显问题，如裁剪不当、背景杂乱 |
| 1-4  | 较差 | 严重构图失误，主体偏移或画面失衡 |

### 表情 (Expression)
| 分数 | 等级 | 描述 |
|------|------|------|
| 9-10 | 优秀 | 情绪到位、眼神有光、表情自然生动 |
| 7-8  | 良好 | 表情自然，有情绪但不够突出 |
| 5-6  | 一般 | 表情平淡或略显僵硬 |
| 1-4  | 较差 | 表情失控、闭眼、或情绪不合场景 |

### 动作 (Pose)
| 分数 | 等级 | 描述 |
|------|------|------|
| 9-10 | 优秀 | 姿态优雅、肢体语言流畅、与场景完美融合 |
| 7-8  | 良好 | 姿态自然，动作协调 |
| 5-6  | 一般 | 姿态略显僵硬或动作不自然 |
| 1-4  | 较差 | 姿态别扭、动作生硬或裁剪不当 |

## 技术细节

- **基础模型**: EfficientNet-B3（ImageNet 预训练）
- **输出维度**: 4 个回归值（构图、表情、动作、总分）
- **损失函数**: 多任务 MSE（各维度可配置权重）
- **总分计算**: 构图 40% + 表情 35% + 动作 25%

## 进阶使用

### 调整总分权重

编辑 `src/model.py` 中的 `overall_head`：
```python
# 修改三个维度对总分的贡献比例
overall = composition * 0.4 + expression * 0.35 + pose * 0.25
```

### 调整损失权重

编辑 `src/train.py` 中的 `MultiDimLoss`：
```python
self.weights = {
    'composition': 1.0,
    'expression': 1.5,  # 更重视表情
    'pose': 1.0,
    'overall': 2.0
}
```

## 许可证

MIT License

## 作者

MX-VI - 人像摄影师
