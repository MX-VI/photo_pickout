# 照片评分模型

此目录用于保存训练好的模型检查点。

## 文件说明

| 文件名 | 说明 |
|--------|------|
| `best_model.pth` | 验证集上表现最好的模型（推荐使用） |
| `checkpoint_epoch_N.pth` | 第 N 个 epoch 的检查点 |

## 模型加载示例

```python
import torch
from src.model import PhotoScorer

# 加载检查点
checkpoint = torch.load('models/best_model.pth')
config = checkpoint['config']

# 创建模型
model = PhotoScorer(
    model_name=config['model']['name'],
    num_classes=config['model']['num_classes'],
    dropout=config['model']['dropout']
)

# 加载权重
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 查看训练信息
print(f"训练轮数: {checkpoint['epoch']}")
print(f"验证损失: {checkpoint['best_val_loss']:.4f}")
print(f"验证 MAE: {checkpoint['val_mae']:.4f}")
```
