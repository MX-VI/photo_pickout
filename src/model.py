"""
模型定义模块
使用 EfficientNet 作为骨干网络，输出 1-10 分的回归预测
"""

import torch
import torch.nn as nn
import timm


class PhotoScorer(nn.Module):
    """
    照片评分模型
    基于预训练的 CNN 架构，输出 1-10 分的连续分数
    """
    
    def __init__(self, model_name='efficientnet_b3', num_classes=1, 
                 pretrained=True, dropout=0.3):
        super(PhotoScorer, self).__init__()
        
        # 使用 timm 加载预训练模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 去掉最后的分类层
            global_pool='avg'
        )
        
        # 获取特征维度
        feature_dim = self.backbone.num_features
        
        # 自定义头部：回归预测
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # 输出 0-1，再映射到 1-10
        )
        
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        # 回归预测并映射到 1-10 分
        score = self.regressor(features)
        score = score * 9 + 1  # 将 0-1 映射到 1-10
        return score


def get_model(config):
    """
    根据配置创建模型
    
    Args:
        config: 配置字典
    
    Returns:
        model: PhotoScorer 实例
    """
    model = PhotoScorer(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    return model


if __name__ == "__main__":
    # 测试模型
    model = PhotoScorer()
    x = torch.randn(2, 3, 300, 300)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: {output.min().item():.2f} - {output.max().item():.2f}")
