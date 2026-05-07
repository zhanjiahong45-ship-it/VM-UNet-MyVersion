# 病变衰减模式可视化工具

## 📋 概述

基于你现有代码中的 `faint_lesion_augmentor.py` 实现的病变衰减模式可视化工具，支持以下4种模式的展示和分析：

### 🔍 四种衰减模式

1. **弥漫型衰减 (Diffuse)**
   - 单个不规则blob，完全衰减
   - 边界模糊，均匀衰减效果
   - 模拟病灶整体淡化

2. **边界型衰减 (Boundary/Donut)**
   - 外环衰减，中心相对保留
   - 形成"甜甜圈"形状
   - 模拟边缘模糊但中心清晰的病灶

3. **多中心/碎片型衰减 (Multifocal)**
   - 多个不连续的衰减区域
   - 模拟多发性病灶
   - 展示碎片化的衰减模式

4. **部分型衰减 (Partial)**
   - 不对称部分衰减
   - 病灶局部淡化
   - 模拟不规则的部分衰减

## 🚀 使用方法

### 方法1：快速演示（推荐新手）

```bash
# 单张图像演示
python run_visualization_demo.py --input inputs/test.jpg --output pattern_demo.png

# 批量处理整个目录
python run_visualization_demo.py --input inputs/ --batch --output pattern_results/
```

### 方法2：完整分析（包含模型预测）

```bash
# 单图详细分析（包含模型预测和特征分析）
python visualize_lesion_patterns.py --input_dir inputs/ --mode single --output_dir detailed_analysis/

# 批量对比仪表板
python visualize_lesion_patterns.py --input_dir inputs/ --mode batch --output_dir comparison_dashboard/
```

### 方法3：自定义模型权重

```bash
python visualize_lesion_patterns.py \
    --model_path /path/to/your/checkpoint.pth \
    --input_dir inputs/ \
    --mode single \
    --output_dir results/
```

## 📊 输出说明

### 快速演示输出
- **单图模式**：生成一张6宫格图像，展示原始图像+4种衰减模式
- **批量模式**：为每张输入图像生成对应的分析图像

### 完整分析输出
- **单图详细分析**：
  - 第一行：原始图像+4种模式的可视化
  - 第二行：模型预测概率图
  - 第三行：特征分析（面积占比、轮廓数量、紧致度等）

- **批量对比仪表板**：
  - 多张图像的并排对比
  - 便于观察不同图像在同一模式下的表现

## 🛠️ 参数说明

### run_visualization_demo.py

```bash
--input        # 输入图像路径或目录
--mask         # mask路径（可选，不提供则自动生成示例mask）
--output       # 输出路径
--batch        # 启用批量处理模式
```

### visualize_lesion_patterns.py

```bash
--model_path   # 模型权重路径
--input_dir    # 输入图像目录
--mask_dir     # mask目录（可选）
--output_dir   # 输出目录
--mode         # single(详细分析) 或 batch(批量对比)
```

## 📈 特征分析

可视化工具会自动计算以下特征指标：

1. **面积占比**：病灶区域占整个图像的比例
2. **轮廓数量**：反映病灶的碎片化程度
3. **紧致度**：圆形度指标，越接近1越接近圆形
4. **描述文本**：针对不同模式的专业描述

## 🎯 应用场景

1. **数据增强预览**：查看不同衰减模式的效果
2. **模型鲁棒性测试**：评估模型对不同模式的预测能力
3. **论文配图**：生成高质量的对比图像
4. **临床教学**：展示不同类型病变的特征

## 💡 提示

- 对于没有mask的图像，工具会自动生成示例mask（中心椭圆）
- 建议使用高分辨率输入图像以获得更好的可视化效果
- 批量处理时建议每次处理不超过10张图像，避免内存不足
- 输出图像都是高分辨率PNG格式，适合论文使用

## 📁 依赖要求

```python
opencv-python
matplotlib
numpy
torch
PIL
```

## 🔧 故障排除

1. **模型加载失败**：检查 `--model_path` 是否正确
2. **图像读取错误**：确认图像格式为 jpg/jpeg/png/bmp
3. **内存不足**：减少批量处理的图像数量
4. **中文显示问题**：在某些系统上可能需要配置中文字体

## 📞 技术支持

基于现有代码库中的以下文件实现：
- `faint_lesion_augmentor.py`：衰减模式生成器
- `predict.py`：模型预测逻辑
- `utils.py`：工具函数

如有问题，请检查这些文件是否正确配置。