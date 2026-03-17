"""
多维度照片评分模型
输出：构图、表情、身体动作三个维度 + 总分
"""

import torch
import torch.nn as nn
import timm


class PhotoScorerMultiDim(nn.Module):
    """
    多维度照片评分模型
    
    输出维度：
    - composition: 构图评分 (1-10)
    - expression: 表情评分 (1-10)  
    - pose: 身体动作评分 (1-10)
    - overall: 综合总分 (1-10)
    """
    
    def __init__(self, model_name='efficientnet_b3', pretrained=True, dropout=0.3):
        super(PhotoScorerMultiDim, self).__init__()
        
        # 共享骨干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        feature_dim = self.backbone.num_features
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 各维度的独立预测头
        self.composition_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.expression_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.pose_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 综合评分头（融合三个维度）
        self.overall_head = nn.Sequential(
            nn.Linear(512 + 3, 128),  # 512维特征 + 3个维度分数
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        
        # 共享特征
        shared_features = self.shared(features)
        
        # 各维度预测 (0-1，后续映射到1-10)
        composition = self.composition_head(shared_features)
        expression = self.expression_head(shared_features)
        pose = self.pose_head(shared_features)
        
        # 将三个维度分数拼接，用于综合评分
        dims = torch.cat([composition, expression, pose], dim=1)
        
        # 综合评分
        combined = torch.cat([shared_features, dims], dim=1)
        overall = self.overall_head(combined)
        
        # 映射到 1-10 分
        composition = composition * 9 + 1
        expression = expression * 9 + 1
        pose = pose * 9 + 1
        overall = overall * 9 + 1
        
        return {
            'composition': composition,
            'expression': expression,
            'pose': pose,
            'overall': overall
        }


def get_model(config):
    """根据配置创建模型"""
    model = PhotoScorerMultiDim(
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    return model


if __name__ == "__main__":
    # 测试模型
    model = PhotoScorerMultiDim()
    x = torch.randn(2, 3, 300, 300)
    output = model(x)
    print(f"Input shape: {x.shape}")
    for key, value in output.items():
        print(f"{key}: {value.shape}, range: {value.min().item():.2f} - {value.max().item():.2f}")
