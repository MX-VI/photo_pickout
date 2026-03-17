"""
数据集加载模块
支持从 good/bad 文件夹加载照片，并自动分配分数标签
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml


class PhotoDataset(Dataset):
    """
    照片评分数据集
    从 good 和 bad 文件夹加载图片，自动分配分数标签
    """
    
    def __init__(self, good_dir, bad_dir, img_size=300, 
                 score_good=(7.0, 10.0), score_bad=(1.0, 6.0),
                 transform=None, augmentation=False):
        """
        Args:
            good_dir: 高质量照片目录
            bad_dir: 低质量照片目录
            img_size: 图像尺寸
            score_good: good 照片分数范围 (min, max)
            score_bad: bad 照片分数范围 (min, max)
            transform: 自定义变换
            augmentation: 是否使用数据增强
        """
        self.img_size = img_size
        self.samples = []
        
        # 支持的图片格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # 加载 good 照片
        if os.path.exists(good_dir):
            for img_path in Path(good_dir).glob('*'):
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append({
                        'path': str(img_path),
                        'score_range': score_good,
                        'label': 'good'
                    })
        
        # 加载 bad 照片
        if os.path.exists(bad_dir):
            for img_path in Path(bad_dir).glob('*'):
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append({
                        'path': str(img_path),
                        'score_range': score_bad,
                        'label': 'bad'
                    })
        
        print(f"Loaded {len(self.samples)} images")
        print(f"  - Good samples: {len([s for s in self.samples if s['label'] == 'good'])}")
        print(f"  - Bad samples: {len([s for s in self.samples if s['label'] == 'bad'])}")
        
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
        
        # 生成分数（在范围内随机采样，增加鲁棒性）
        score = random.uniform(sample['score_range'][0], sample['score_range'][1])
        score = torch.tensor([score], dtype=torch.float32)
        
        return image, score


def get_data_loaders(config):
    """
    创建训练和验证数据加载器
    
    Args:
        config: 配置字典
    
    Returns:
        train_loader, val_loader
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 创建完整数据集
    full_dataset = PhotoDataset(
        good_dir=data_cfg['good_dir'],
        bad_dir=data_cfg['bad_dir'],
        img_size=data_cfg['img_size'],
        score_good=(data_cfg['score_mapping']['good_min'], 
                   data_cfg['score_mapping']['good_max']),
        score_bad=(data_cfg['score_mapping']['bad_min'], 
                  data_cfg['score_mapping']['bad_max']),
        augmentation=True
    )
    
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
    dataset = PhotoDataset(
        good_dir='./data/good',
        bad_dir='./data/bad',
        img_size=300,
        augmentation=True
    )
    
    if len(dataset) > 0:
        image, score = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Score: {score.item():.2f}")
    else:
        print("No images found. Please add photos to ./data/good and ./data/bad")
