"""
多维度标注工具
帮助用户给照片打上构图、表情、动作三个维度的分数
"""

import os
import json
import argparse
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox


class PhotoAnnotator:
    """交互式照片标注工具"""
    
    def __init__(self, image_dir, output_file='data/annotations.json'):
        self.image_dir = Path(image_dir)
        self.output_file = Path(output_file)
        self.annotations = {}
        
        # 加载已有标注
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            print(f"Loaded {len(self.annotations)} existing annotations")
        
        # 获取所有图片
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.images = [f for f in self.image_dir.glob('*') 
                      if f.suffix.lower() in valid_extensions]
        self.images.sort()
        
        # 过滤已标注的
        self.pending_images = [img for img in self.images 
                              if img.name not in self.annotations]
        
        self.current_idx = 0
        self.current_scores = {'composition': 7.0, 'expression': 7.0, 'pose': 7.0}
        
        print(f"Found {len(self.images)} images, {len(self.pending_images)} pending")
    
    def save_annotation(self):
        """保存当前标注"""
        if self.current_idx >= len(self.pending_images):
            return
        
        img_name = self.pending_images[self.current_idx].name
        
        # 计算总分（加权平均）
        overall = (
            self.current_scores['composition'] * 0.4 +
            self.current_scores['expression'] * 0.35 +
            self.current_scores['pose'] * 0.25
        )
        
        self.annotations[img_name] = {
            'composition': round(self.current_scores['composition'], 1),
            'expression': round(self.current_scores['expression'], 1),
            'pose': round(self.current_scores['pose'], 1),
            'overall': round(overall, 1)
        }
        
        # 保存到文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        
        print(f"Saved: {img_name}")
        print(f"  构图: {self.current_scores['composition']:.1f}")
        print(f"  表情: {self.current_scores['expression']:.1f}")
        print(f"  动作: {self.current_scores['pose']:.1f}")
        print(f"  总分: {overall:.1f}")
    
    def next_image(self, event=None):
        """下一张图片"""
        self.save_annotation()
        self.current_idx += 1
        
        if self.current_idx >= len(self.pending_images):
            print("\n✅ 所有图片标注完成！")
            plt.close()
            return
        
        self.current_scores = {'composition': 7.0, 'expression': 7.0, 'pose': 7.0}
        self.show_image()
    
    def skip_image(self, event=None):
        """跳过当前图片"""
        self.current_idx += 1
        if self.current_idx >= len(self.pending_images):
            print("\n✅ 所有图片处理完成！")
            plt.close()
            return
        self.show_image()
    
    def set_score(self, dimension, value):
        """设置某个维度的分数"""
        try:
            score = float(value)
            score = max(1.0, min(10.0, score))
            self.current_scores[dimension] = score
        except ValueError:
            pass
    
    def show_image(self):
        """显示当前图片"""
        if self.current_idx >= len(self.pending_images):
            return
        
        img_path = self.pending_images[self.current_idx]
        
        # 清除当前图像
        self.ax_img.clear()
        
        # 显示图片
        img = Image.open(img_path)
        self.ax_img.imshow(img)
        self.ax_img.axis('off')
        self.ax_img.set_title(
            f"[{self.current_idx + 1}/{len(self.pending_images)}] {img_path.name}\n"
            f"构图: {self.current_scores['composition']:.1f} | "
            f"表情: {self.current_scores['expression']:.1f} | "
            f"动作: {self.current_scores['pose']:.1f}",
            fontsize=12
        )
        
        # 更新输入框
        self.text_comp.set_val(str(self.current_scores['composition']))
        self.text_expr.set_val(str(self.current_scores['expression']))
        self.text_pose.set_val(str(self.current_scores['pose']))
        
        self.fig.canvas.draw()
    
    def run(self):
        """运行标注工具"""
        if not self.pending_images:
            print("没有需要标注的图片！")
            return
        
        # 创建图形界面
        self.fig, self.ax_img = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.15)
        
        # 创建输入框
        ax_comp = plt.axes([0.15, 0.05, 0.1, 0.04])
        ax_expr = plt.axes([0.35, 0.05, 0.1, 0.04])
        ax_pose = plt.axes([0.55, 0.05, 0.1, 0.04])
        
        self.text_comp = TextBox(ax_comp, '构图', initial='7.0')
        self.text_expr = TextBox(ax_expr, '表情', initial='7.0')
        self.text_pose = TextBox(ax_pose, '动作', initial='7.0')
        
        self.text_comp.on_submit(lambda x: self.set_score('composition', x))
        self.text_expr.on_submit(lambda x: self.set_score('expression', x))
        self.text_pose.on_submit(lambda x: self.set_score('pose', x))
        
        # 创建按钮
        ax_next = plt.axes([0.72, 0.05, 0.1, 0.05])
        ax_skip = plt.axes([0.85, 0.05, 0.1, 0.05])
        
        btn_next = Button(ax_next, '保存并下一张')
        btn_skip = Button(ax_skip, '跳过')
        
        btn_next.on_clicked(self.next_image)
        btn_skip.on_clicked(self.skip_image)
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 显示第一张图片
        self.show_image()
        
        print("\n标注工具启动！")
        print("使用方法：")
        print("  1. 在输入框中输入 1-10 的分数")
        print("  2. 点击 '保存并下一张' 或按 Enter")
        print("  3. 点击 '跳过' 跳过当前图片")
        print("  快捷键：1-9 快速设置构图分数")
        
        plt.show()
    
    def on_key(self, event):
        """键盘事件处理"""
        if event.key in '123456789':
            self.current_scores['composition'] = float(event.key)
            self.text_comp.set_val(event.key)
            self.show_image()
        elif event.key == 'enter':
            self.next_image()


def quick_annotate(image_dir, output_file='data/annotations.json'):
    """快速标注模式 - 命令行交互"""
    image_dir = Path(image_dir)
    output_file = Path(output_file)
    
    # 加载已有标注
    annotations = {}
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    
    # 获取待标注图片
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in image_dir.glob('*') if f.suffix.lower() in valid_extensions]
    images.sort()
    
    pending = [img for img in images if img.name not in annotations]
    
    print(f"Found {len(images)} images, {len(pending)} pending")
    print("\n快速标注模式")
    print("输入格式: 构图分数,表情分数,动作分数 (如: 8,7,6)")
    print("输入 's' 跳过, 'q' 退出\n")
    
    for img_path in pending:
        print(f"\n[{len(annotations)+1}/{len(images)}] {img_path.name}")
        
        # 显示图片（如果有图像查看器）
        try:
            img = Image.open(img_path)
            img.thumbnail((400, 400))
            img.show()
        except:
            pass
        
        while True:
            user_input = input("评分 (构,表,动): ").strip()
            
            if user_input.lower() == 'q':
                print("保存并退出...")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, ensure_ascii=False, indent=2)
                return
            
            if user_input.lower() == 's':
                print("跳过")
                break
            
            try:
                parts = user_input.split(',')
                if len(parts) == 3:
                    comp = float(parts[0])
                    expr = float(parts[1])
                    pose = float(parts[2])
                    
                    # 验证范围
                    for val, name in [(comp, '构图'), (expr, '表情'), (pose, '动作')]:
                        if not 1 <= val <= 10:
                            print(f"{name}分数 {val} 超出范围 (1-10)")
                            raise ValueError
                    
                    overall = comp * 0.4 + expr * 0.35 + pose * 0.25
                    
                    annotations[img_path.name] = {
                        'composition': round(comp, 1),
                        'expression': round(expr, 1),
                        'pose': round(pose, 1),
                        'overall': round(overall, 1)
                    }
                    
                    # 保存
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(annotations, f, ensure_ascii=False, indent=2)
                    
                    print(f"  已保存: 构{comp:.1f} 表{expr:.1f} 动{pose:.1f} 总{overall:.1f}")
                    break
                    
            except ValueError:
                print("格式错误，请重新输入 (如: 8,7,6)")
    
    print(f"\n✅ 标注完成！共标注 {len(annotations)} 张图片")
    print(f"标注文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='照片多维度标注工具')
    parser.add_argument('image_dir', type=str, help='图片文件夹路径')
    parser.add_argument('--output', type=str, default='data/annotations.json',
                       help='标注文件输出路径')
    parser.add_argument('--gui', action='store_true', help='使用图形界面（需要桌面环境）')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"错误：文件夹不存在 {args.image_dir}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.gui:
        # 图形界面模式
        annotator = PhotoAnnotator(args.image_dir, args.output)
        annotator.run()
    else:
        # 命令行模式
        quick_annotate(args.image_dir, args.output)


if __name__ == "__main__":
    main()
