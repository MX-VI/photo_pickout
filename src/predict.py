"""
照片评分推理脚本
支持单张照片评分和批量评分
"""

import os
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import PhotoScorer


def load_model(checkpoint_path, device='cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = PhotoScorer(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,  # 推理时不需要预训练权重
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
    image = image.unsqueeze(0)  # 添加 batch 维度
    
    return image


def predict_single(model, image_path, img_size, device):
    """预测单张照片分数"""
    image = preprocess_image(image_path, img_size)
    image = image.to(device)
    
    with torch.no_grad():
        score = model(image)
    
    return score.item()


def predict_batch(model, folder_path, img_size, device):
    """批量预测文件夹内所有照片"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in Path(folder_path).glob('*') 
                   if f.suffix.lower() in valid_extensions]
    
    results = []
    
    for img_path in tqdm(image_files, desc="Scoring photos"):
        try:
            score = predict_single(model, str(img_path), img_size, device)
            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'score': round(score, 2),
                'grade': score_to_grade(score)
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
    
    # 1. 分数分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(df['score'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Photo Score Distribution')
    plt.savefig(f'{output_dir}/score_distribution.png', dpi=150)
    plt.close()
    
    # 2. 等级分布饼图
    grade_counts = df['grade'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%')
    plt.title('Photo Grade Distribution')
    plt.savefig(f'{output_dir}/grade_distribution.png', dpi=150)
    plt.close()
    
    # 3. 保存详细结果
    df_sorted = df.sort_values('score', ascending=False)
    df_sorted.to_csv(f'{output_dir}/scoring_results.csv', index=False, encoding='utf-8-sig')
    
    # 4. 生成汇总统计
    summary = {
        'total_photos': len(df),
        'average_score': round(df['score'].mean(), 2),
        'median_score': round(df['score'].median(), 2),
        'min_score': round(df['score'].min(), 2),
        'max_score': round(df['score'].max(), 2),
        'std_score': round(df['score'].std(), 2),
        'excellent_count': len(df[df['grade'] == '优秀']),
        'good_count': len(df[df['grade'] == '良好']),
        'average_count': len(df[df['grade'] == '一般']),
        'poor_count': len(df[df['grade'] == '较差'])
    }
    
    # 保存汇总
    with open(f'{output_dir}/summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 40 + "\n")
        f.write("       照片评分汇总报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"总照片数: {summary['total_photos']}\n")
        f.write(f"平均分: {summary['average_score']}\n")
        f.write(f"中位数: {summary['median_score']}\n")
        f.write(f"最低分: {summary['min_score']}\n")
        f.write(f"最高分: {summary['max_score']}\n")
        f.write(f"标准差: {summary['std_score']}\n\n")
        f.write("等级分布:\n")
        f.write(f"  优秀 (9-10分): {summary['excellent_count']} 张\n")
        f.write(f"  良好 (7-8分): {summary['good_count']} 张\n")
        f.write(f"  一般 (5-6分): {summary['average_count']} 张\n")
        f.write(f"  较差 (1-4分): {summary['poor_count']} 张\n")
        f.write("\n" + "=" * 40 + "\n")
        f.write("详细结果已保存至: scoring_results.csv\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Photo Pickout - 照片评分')
    parser.add_argument('--model', type=str, default='./models/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--image', type=str, default=None,
                       help='单张照片路径')
    parser.add_argument('--folder', type=str, default=None,
                       help='照片文件夹路径（批量评分）')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--sort', action='store_true',
                       help='按分数排序输出')
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
        
        score = predict_single(model, args.image, img_size, device)
        grade = score_to_grade(score)
        
        print(f"\n{'='*40}")
        print(f"照片: {args.image}")
        print(f"评分: {score:.2f} / 10")
        print(f"等级: {grade}")
        print(f"{'='*40}\n")
    
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
            results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # 只保留前 N 张
        if args.top:
            results = results[:args.top]
        
        # 生成报告
        summary = generate_report(results, args.output)
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"              评分完成！共 {summary['total_photos']} 张照片")
        print(f"{'='*60}")
        print(f"平均分: {summary['average_score']} | 中位数: {summary['median_score']}")
        print(f"范围: {summary['min_score']} - {summary['max_score']}")
        print(f"\n等级分布:")
        print(f"  优秀: {summary['excellent_count']} | 良好: {summary['good_count']} | "
              f"一般: {summary['average_count']} | 较差: {summary['poor_count']}")
        print(f"\n结果已保存至: {args.output}/")
        print(f"{'='*60}\n")
        
        # 打印前 10 名
        print("🏆 高分照片 TOP 10:")
        for i, r in enumerate(results[:10], 1):
            print(f"  {i:2d}. {r['score']:5.2f}分 | {r['filename'][:40]}")
    
    else:
        print("请指定 --image 或 --folder 参数")
        print("示例:")
        print("  python src/predict.py --image photo.jpg")
        print("  python src/predict.py --folder ./photos --sort --top 20")


if __name__ == "__main__":
    main()
