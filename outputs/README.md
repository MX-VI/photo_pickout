# 输出目录

此目录用于保存推理结果和可视化报告。

## 目录结构

```
outputs/
├── plots/                    # 图表
│   ├── score_distribution.png    # 分数分布直方图
│   └── grade_distribution.png    # 等级分布饼图
├── reports/                 # 报告文件
│   ├── scoring_results.csv      # 详细评分结果
│   └── summary.txt              # 汇总统计
└── [其他自定义输出]
```

## 输出文件说明

### scoring_results.csv
包含每张照片的详细评分信息：
- `filename`: 文件名
- `filepath`: 完整路径
- `score`: 评分（1-10）
- `grade`: 等级（优秀/良好/一般/较差）

### summary.txt
汇总统计信息，包括：
- 总照片数
- 平均分/中位数/最高分/最低分
- 各等级照片数量

## 使用示例

```bash
# 批量评分并生成报告
python src/predict.py --folder ./photos --output ./outputs

# 查看结果
cat outputs/summary.txt
cat outputs/scoring_results.csv
```
