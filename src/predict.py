"""
多维度照片评分推理脚本
输出：构图、表情、身体动作三个维度 + 总分
"""

import os
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import PhotoScorerMultiDim


def load_model(checkpoint_path, device='cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = PhotoScorerMultiDim(
        model_name=config['model']['name'],
        pretrained=False,
        dropout=config['model']['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def preprocess_image(image_path, img_size):
    """预处理单张图片"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    
    return image


def predict_single(model, image_path, img_size, device):
    """预测单张照片的多维度分数"""
    image = preprocess_image(image_path, img_size)
    image = image.to(device)
    
    with torch.no_grad():
        scores = model(image)
    
    return {k: v.item() for k, v in scores.items()}


def predict_batch(model, folder_path, img_size, device):
    """批量预测文件夹内所有照片"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in Path(folder_path).glob('*') 
                   if f.suffix.lower() in valid_extensions]
    
    results = []
    
    for img_path in tqdm(image_files, desc="Scoring photos"):
        try:
            scores = predict_single(model, str(img_path), img_size, device)
            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'composition': round(scores['composition'], 2),
                'expression': round(scores['expression'], 2),
                'pose': round(scores['pose'], 2),
                'overall': round(scores['overall'], 2),
                'composition_grade': score_to_grade(scores['composition']),
                'expression_grade': score_to_grade(scores['expression']),
                'pose_grade': score_to_grade(scores['pose']),
                'overall_grade': score_to_grade(scores['overall'])
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return results


def score_to_grade(score):
    """将分数转换为等级"""
    if score >= 9:
        return '优秀'
    elif score >= 7:
        return '良好'
    elif score >= 5:
        return '一般'
    else:
        return '较差'


def generate_report(results, output_dir):
    """生成可视化报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # 1. 各维度分数分布图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    dimensions = ['composition', 'expression', 'pose', 'overall']
    titles = ['构图评分分布', '表情评分分布', '动作评分分布', '总分分布']
    
    for ax, dim, title in zip(axes.flat, dimensions, titles):
        ax.hist(df[dim], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.axvline(df[dim].mean(), color='red', linestyle='--', label=f'Mean: {df[dim].mean():.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distributions.png', dpi=150)
    plt.close()
    
    # 2. 各维度平均分对比
    plt.figure(figsize=(10, 6))
    means = [df[d].mean() for d in dimensions]
    plt.bar(dimensions, means, color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c'])
    plt.ylabel('Average Score')
    plt.title('各维度平均分对比')
    plt.ylim(0, 10)
    for i, v in enumerate(means):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    plt.savefig(f'{output_dir}/dimension_comparison.png', dpi=150)
    plt.close()
    
    # 3. 保存详细结果
    df_sorted = df.sort_values('overall', ascending=False)
    df_sorted.to_csv(f'{output_dir}/scoring_results.csv', index=False, encoding='utf-8-sig')
    
    # 4. 生成汇总统计
    summary = {
        'total_photos': len(df),
        'dimensions': {}
    }
    
    for dim in dimensions:
        summary['dimensions'][dim] = {
            'mean': round(df[dim].mean(), 2),
            'median': round(df[dim].median(), 2),
            'min': round(df[dim].min(), 2),
            'max': round(df[dim].max(), 2),
            'std': round(df[dim].std(), 2)
        }
    
    # 保存汇总
    with open(f'{output_dir}/summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("       多维度照片评分汇总报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总照片数: {summary['total_photos']}\n\n")
        
        dim_names = {'composition': '构图', 'expression': '表情', 'pose': '动作', 'overall': '总分'}
        for dim, name in dim_names.items():
            stats = summary['dimensions'][dim]
            f.write(f"【{name}】\n")
            f.write(f"  平均分: {stats['mean']} | 中位数: {stats['median']}\n")
            f.write(f"  最低分: {stats['min']} | 最高分: {stats['max']} | 标准差: {stats['std']}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("详细结果已保存至: scoring_results.csv\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Photo Pickout - 多维度照片评分')
    parser.add_argument('--model', type=str, default='./models/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--image', type=str, default=None,
                       help='单张照片路径')
    parser.add_argument('--folder', type=str, default=None,
                       help='照片文件夹路径（批量评分）')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--sort', action='store_true',
                       help='按总分排序输出')
    parser.add_argument('--top', type=int, default=None,
                       help='只显示前 N 张最高分照片')
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误：模型文件不存在 {args.model}")
        print("请先运行训练脚本：python src/train.py")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading model...")
    model, config = load_model(args.model, device)
    img_size = config['data']['img_size']
    
    # 单张照片评分
    if args.image:
        if not os.path.exists(args.image):
            print(f"错误：图片不存在 {args.image}")
            return
        
        scores = predict_single(model, args.image, img_size, device)
        
        print(f"\n{'='*50}")
        print(f"照片: {args.image}")
        print(f"{'='*50}")
        print(f"构图评分: {scores['composition']:.2f}/10  [{score_to_grade(scores['composition'])}]")
        print(f"表情评分: {scores['expression']:.2f}/10  [{score_to_grade(scores['expression'])}]")
        print(f"动作评分: {scores['pose']:.2f}/10      [{score_to_grade(scores['pose'])}]")
        print(f"{'-'*50}")
        print(f"综合总分: {scores['overall']:.2f}/10  [{score_to_grade(scores['overall'])}]")
        print(f"{'='*50}\n")
    
    # 批量评分
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"错误：文件夹不存在 {args.folder}")
            return
        
        print(f"\n开始评分文件夹: {args.folder}")
        results = predict_batch(model, args.folder, img_size, device)
        
        if not results:
            print("没有找到图片文件")
            return
        
        # 排序
        if args.sort:
            results = sorted(results, key=lambda x: x['overall'], reverse=True)
        
        # 只保留前 N 张
        if args.top:
            results = results[:args.top]
        
        # 生成报告
        summary = generate_report(results, args.output)
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"              评分完成！共 {summary['total_photos']} 张照片")
        print(f"{'='*60}")
        
        dims = summary['dimensions']
        print(f"构图: {dims['composition']['mean']:.2f} | "
              f"表情: {dims['expression']['mean']:.2f} | "
              f"动作: {dims['pose']['mean']:.2f} | "
              f"总分: {dims['overall']['mean']:.2f}")
        print(f"\n结果已保存至: {args.output}/")
        print(f"{'='*60}\n")
        
        # 打印前 10 名
        print("🏆 高分照片 TOP 10:")
        for i, r in enumerate(results[:10], 1):
            print(f"  {i:2d}. 总分{r['overall']:5.2f} | "
                  f"构{r['composition']:4.1f} 表{r['expression']:4.1f} 动{r['pose']:4.1f} | "
                  f"{r['filename'][:30]}")
    
    else:
        print("请指定 --image 或 --folder 参数")
        print("示例:")
        print("  python src/predict.py --image photo.jpg")
        print("  python src/predict.py --folder ./photos --sort --top 20")


if __name__ == "__main__":
    main()
