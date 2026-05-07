# Springer论文图使用说明

## 📊 生成的图像

**文件名：** `springer_paper_figure.png`
**尺寸：** 2180×520 pixels
**打印尺寸：** 18.5cm × 4.4cm @ 300dpi
**布局：** 1×5横向排列

## 🔍 图像内容

该图展示了5种病变衰减模式：
- **(a)** Original - 原始图像
- **(b)** Diffuse - 弥漫型衰减
- **(c)** Boundary - 边界型衰减
- **(d)** Multifocal - 多中心型衰减
- **(e)** Partial - 部分型衰减

## 📝 LaTeX使用示例

### 单栏图（推荐）
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{springer_paper_figure.png}
\caption{Demonstration of different lesion attenuation patterns: (a) original image, (b) diffuse attenuation, (c) boundary attenuation, (d) multifocal attenuation, (e) partial attenuation. Red contours indicate lesion boundaries.}
\label{fig:lesion_patterns}
\end{figure}
```

### 双栏图（如果期刊支持）
```latex
\begin{figure*}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{springer_paper_figure.png}
\caption{Demonstration of different lesion attenuation patterns for skin lesion analysis.}
\label{fig:lesion_patterns}
\end{figure*}
```

### 控制高度版本
```latex
\begin{figure}[htbp]
\centering
\includegraphics[height=4cm,keepaspectratio]{springer_paper_figure.png}
\caption{Lesion pattern variations showing different attenuation modes.}
\label{fig:lesion_patterns}
\end{figure}
```

## 📐 Springer图片要求

✅ **已满足的要求：**
- 分辨率：300dpi（推荐：300-600dpi）
- 格式：PNG（支持透明背景）
- 尺寸：适合单栏或双栏排版
- 标签：清晰的子图标签 (a)-(e)
- 字体：清晰易读

## 🔧 如果需要调整尺寸

### 更高的分辨率（600dpi）
```bash
python generate_paper_figure.py --output springer_600dpi.png --dpi 600
```

### 更大的单栏版本
```bash
python generate_paper_figure.py --output springer_single_column.png --dpi 300
# 然后在LaTeX中使用：
# \includegraphics[width=\columnwidth]{springer_single_column.png}
```

### 多行版本（2-3行）
```bash
python generate_paper_figure.py --multirow --num_rows 2 --output springer_multirow.png
```

## 📄 图注建议

### 简短版
```latex
\caption{Lesion attenuation patterns: (a) original, (b) diffuse, (c) boundary, (d) multifocal, (e) partial.}
```

### 详细版
```latex
\caption{Visualization of five lesion attenuation patterns used in our data augmentation strategy. (a) Original lesion image; (b) Diffuse attenuation showing uniform fading across the lesion; (c) Boundary attenuation demonstrating edge-blurring with preserved center; (d) Multifocal attenuation presenting scattered fragmented patterns; (e) Partial attenuation displaying asymmetric regional fading. Red contours delineate lesion boundaries.}
```

## 🎨 颜色和打印

- **在线版本：** 使用PNG格式，颜色鲜艳
- **打印版本：** PNG格式在论文打印中效果很好
- **黑白打印：** 红色轮廓在灰度图中仍清晰可见

## 📧 投稿建议

1. **检查期刊要求：** 不同Springer期刊可能有特定要求
2. **保持高分辨率：** 建议提交300dpi以上版本
3. **文件大小：** 当前文件大小适中，通常符合投稿要求
4. **文件命名：** 建议使用描述性文件名，如 `fig1_lesion_patterns.png`

## 🔗 在文中引用

```latex
As shown in Figure~\ref{fig:lesion_patterns}, we demonstrate five different lesion attenuation patterns...
```

## 💡 其他建议

- 如果需要调整图像内容或布局，修改 `generate_paper_figure.py` 中的参数
- 对于彩色图片，确保期刊支持彩色印刷（或收取额外费用）
- 建议在投稿前查看期刊的具体作者指南