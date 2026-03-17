"""
模型训练脚本
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


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, scores in pbar:
        images = images.to(device)
        scores = scores.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, scores in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            scores = scores.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, scores)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(scores.cpu().numpy())
    
    # 计算 MAE (Mean Absolute Error)
    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)
    mae = torch.abs(predictions - targets).mean().item()
    
    return total_loss / len(val_loader), mae


def main():
    parser = argparse.ArgumentParser(description='Train Photo Pickout Model')
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
    
    if len(train_loader.dataset) == 0:
        print("错误：没有找到训练数据！")
        print(f"请将照片放入 {config['paths']['good_dir']} 和 {config['paths']['bad_dir']}")
        return
    
    # 创建模型
    print("\nCreating model...")
    model = get_model(config)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
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
    print("=" * 50)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch [{epoch+1}/{config['training']['epochs']}]")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | LR: {current_lr:.6f}")
        
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
                'val_mae': val_mae,
                'config': config
            }, best_model_path)
            print(f"  -> Saved best model (Val Loss: {val_loss:.4f})")
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
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best model saved at: {config['paths']['model_save_dir']}/best_model.pth")


if __name__ == "__main__":
    main()
