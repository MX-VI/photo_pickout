"""
多维度数据集加载模块
支持加载构图、表情、动作三个维度的标签
"""

import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml


class PhotoDatasetMultiDim(Dataset):
    """
    多维度照片评分数据集
    
    支持两种标注方式：
    1. 简单分类：good/bad 文件夹（自动分配分数）
    2. 多维度标注：JSON 文件包含 composition, expression, pose 三个维度
    """
    
    def __init__(self, good_dir, bad_dir, annotations_file=None, img_size=300,
                 score_good=(7.0, 10.0), score_bad=(1.0, 6.0),
                 transform=None, augmentation=False):
        """
        Args:
            good_dir: 高质量照片目录
            bad_dir: 低质量照片目录
            annotations_file: 多维度标注文件路径（JSON格式）
            img_size: 图像尺寸
            score_good: good 照片分数范围
            score_bad: bad 照片分数范围
            transform: 自定义变换
            augmentation: 是否使用数据增强
        """
        self.img_size = img_size
        self.samples = []
        self.multi_dim_mode = annotations_file is not None and os.path.exists(annotations_file)
        
        # 加载多维度标注
        self.annotations = {}
        if self.multi_dim_mode:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            print(f"Loaded multi-dimensional annotations: {len(self.annotations)} images")
        
        # 支持的图片格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # 加载照片
        def load_from_dir(directory, default_label):
            if not os.path.exists(directory):
                return
            for img_path in Path(directory).glob('*'):
                if img_path.suffix.lower() in valid_extensions:
                    img_name = img_path.name
                    
                    if self.multi_dim_mode and img_name in self.annotations:
                        # 使用多维度标注
                        anno = self.annotations[img_name]
                        self.samples.append({
                            'path': str(img_path),
                            'composition': anno.get('composition'),
                            'expression': anno.get('expression'),
                            'pose': anno.get('pose'),
                            'overall': anno.get('overall'),
                            'has_multi_dim': True
                        })
                    else:
                        # 使用简单分类标签
                        score_range = score_good if default_label == 'good' else score_bad
                        self.samples.append({
                            'path': str(img_path),
                            'score_range': score_range,
                            'label': default_label,
                            'has_multi_dim': False
                        })
        
        load_from_dir(good_dir, 'good')
        load_from_dir(bad_dir, 'bad')
        
        # 统计
        multi_dim_count = sum(1 for s in self.samples if s['has_multi_dim'])
        simple_count = len(self.samples) - multi_dim_count
        
        print(f"Loaded {len(self.samples)} images:")
        print(f"  - Multi-dimensional labels: {multi_dim_count}")
        print(f"  - Simple labels: {simple_count}")
        
        # 设置变换
        if transform is None:
            if augmentation:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size + 20, img_size + 20)),
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片
        image = Image.open(sample['path']).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if sample['has_multi_dim']:
            # 多维度标注
            composition = torch.tensor([sample['composition']], dtype=torch.float32)
            expression = torch.tensor([sample['expression']], dtype=torch.float32)
            pose = torch.tensor([sample['pose']], dtype=torch.float32)
            overall = torch.tensor([sample['overall']], dtype=torch.float32)
        else:
            # 简单分类：所有维度用相同分数
            score = random.uniform(sample['score_range'][0], sample['score_range'][1])
            composition = torch.tensor([score], dtype=torch.float32)
            expression = torch.tensor([score], dtype=torch.float32)
            pose = torch.tensor([score], dtype=torch.float32)
            overall = torch.tensor([score], dtype=torch.float32)
        
        return image, {
            'composition': composition,
            'expression': expression,
            'pose': pose,
            'overall': overall
        }


def get_data_loaders(config):
    """创建训练和验证数据加载器"""
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 检查是否有标注文件
    annotations_file = data_cfg.get('annotations_file')
    
    # 创建完整数据集
    full_dataset = PhotoDatasetMultiDim(
        good_dir=data_cfg['good_dir'],
        bad_dir=data_cfg['bad_dir'],
        annotations_file=annotations_file,
        img_size=data_cfg['img_size'],
        score_good=(data_cfg['score_mapping']['good_min'], 
                   data_cfg['score_mapping']['good_max']),
        score_bad=(data_cfg['score_mapping']['bad_min'], 
                  data_cfg['score_mapping']['bad_max']),
        augmentation=True
    )
    
    if len(full_dataset) == 0:
        print("警告：没有找到训练数据！")
        return None, None
    
    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * data_cfg['train_split'])
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 验证集不使用数据增强
    val_dataset.dataset.transform = transforms.Compose([
        transforms.Resize((data_cfg['img_size'], data_cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据集
    dataset = PhotoDatasetMultiDim(
        good_dir='./data/good',
        bad_dir='./data/bad',
        img_size=300,
        augmentation=True
    )
    
    if len(dataset) > 0:
        image, scores = dataset[0]
        print(f"Image shape: {image.shape}")
        for key, value in scores.items():
            print(f"{key}: {value.item():.2f}")
    else:
        print("No images found. Please add photos to ./data/good and ./data/bad")
