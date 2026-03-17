# Photo Pickout

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
将你的照片放入对应文件夹：
- `data/good/` - 高质量照片（7-10分）
- `data/bad/` - 低质量照片（1-6分）

### 3. 训练模型
```bash
python src/train.py --epochs 50
```

### 4. 照片评分
```bash
# 单张照片
python src/predict.py --image photo.jpg

# 批量评分
python src/predict.py --folder ./photos --sort --top 20
```

详见 [README.md](README.md)
