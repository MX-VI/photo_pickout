"""
多维度模型训练脚本
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from model import get_model
from dataset import get_data_loaders


class MultiDimLoss(nn.Module):
    """多维度评分损失函数"""
    
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'composition': 1.0,
            'expression': 1.0,
            'pose': 1.0,
            'overall': 2.0  # 总分权重更高
        }
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        loss = 0
        for key in self.weights:
            loss += self.weights[key] * self.mse(predictions[key], targets[key])
        return loss


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    dim_losses = {'composition': 0, 'expression': 0, 'pose': 0, 'overall': 0}
    
    pbar = tqdm(train_loader, desc='Training')
    for images, scores in pbar:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in scores.items()}
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算各维度损失
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 记录各维度损失
        with torch.no_grad():
            for key in dim_losses:
                dim_loss = nn.functional.mse_loss(outputs[key], targets[key])
                dim_losses[key] += dim_loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 平均损失
    for key in dim_losses:
        dim_losses[key] /= len(train_loader)
    
    return total_loss / len(train_loader), dim_losses


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    dim_maes = {'composition': [], 'expression': [], 'pose': [], 'overall': []}
    
    with torch.no_grad():
        for images, scores in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in scores.items()}
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # 记录各维度 MAE
            for key in dim_maes:
                mae = torch.abs(outputs[key] - targets[key]).mean().item()
                dim_maes[key].append(mae)
    
    # 计算平均 MAE
    avg_maes = {k: sum(v) / len(v) for k, v in dim_maes.items()}
    
    return total_loss / len(val_loader), avg_maes


def main():
    parser = argparse.ArgumentParser(description='Train Photo Pickout Multi-Dim Model')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率（覆盖配置文件）')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # 创建输出目录
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据加载器
    print("\nLoading data...")
    train_loader, val_loader = get_data_loaders(config)
    
    if train_loader is None or len(train_loader.dataset) == 0:
        print("错误：没有找到训练数据！")
        print(f"请将照片放入 {config['paths']['good_dir']} 和 {config['paths']['bad_dir']}")
        return
    
    # 创建模型
    print("\nCreating model...")
    model = get_model(config)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = MultiDimLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['lr_scheduler']['T_max'],
        eta_min=config['training']['lr_scheduler']['eta_min']
    )
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"{config['paths']['log_dir']}/run_{timestamp}")
    
    # 早停设置
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_epoch = 0
    
    # 恢复训练
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # 训练循环
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print("=" * 60)
    print("训练维度: 构图 | 表情 | 身体动作 | 总分")
    print("=" * 60)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch [{epoch+1}/{config['training']['epochs']}]")
        
        # 训练
        train_loss, train_dim_losses = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_maes = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for key, value in val_maes.items():
            writer.add_scalar(f'MAE/{key}', value, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"  MAE - 构图: {val_maes['composition']:.3f} | "
              f"表情: {val_maes['expression']:.3f} | "
              f"动作: {val_maes['pose']:.3f} | "
              f"总分: {val_maes['overall']:.3f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = f"{config['paths']['model_save_dir']}/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_maes': val_maes,
                'config': config
            }, best_model_path)
            print(f"  -> Saved best model")
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = f"{config['paths']['model_save_dir']}/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
        
        # 早停检查
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved at: {config['paths']['model_save_dir']}/best_model.pth")


if __name__ == "__main__":
    main()
