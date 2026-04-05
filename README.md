# Milo Color Analyzer

一个基于 Python 的本地图片色彩分析工具，支持：

- 单张图片色彩分析
- 多张图片的组图风格分析
- 浏览器上传与本地预览
- 图表点击放大查看

项目适合做摄影作品、视觉风格、配色倾向的快速分析，不依赖前端构建工具，直接本地运行即可。

## 功能概览

### 单图分析

上传一张或多张图片后，每张图会生成独立分析结果，包括：

- 原图预览
- Lab `a*b*` 色度分布
- 饱和度与亮度关系
- 亮度分布与 RGB 通道直方图
- Brightness / Saturation 图
- Saturation / Brightness 图
- 交互式 3D LAB 模型

并提供：

- 平均亮度
- 平均饱和度
- 平均色相
- 平均颜色
- 像素数

### 组图风格分析

上传一个作品文件夹后，会对整组图片进行聚合分析，输出：

- 整体风格总结
- 风格标签
- 综合色与范围指标
- 九宫格空间色彩分布
- 多种组图可视化图表

当前组图页面包含 8 类主要图表：

- Lab `a*b*` Distribution
- Color Harmony
- Hue / Saturation Heatmap
- Luminance Layers
- Contrast Profile
- Spatial Palette Grid
- Image Fingerprint Map
- Mood Coordinates

所有图表都支持点击放大预览。

## 项目结构

```text
COLOR/
├── color_analysis.py     # 主后端与图表生成逻辑
├── color.py              # 参考分析脚本
├── run_web.sh            # 本地启动脚本
└── web/
    ├── index.html        # 单图分析页面
    ├── collection.html   # 组图分析页面
    └── styles.css        # 页面样式
```

## 运行要求

建议使用 Python 3.10 及以上。

依赖：

- `numpy`
- `matplotlib`
- `Pillow`

可选依赖：

- `scikit-image`

安装示例：

```bash
pip install numpy matplotlib pillow scikit-image
```

## 启动方式

启动本地 Web 应用：

```bash
./run_web.sh
```

默认地址：

```text
http://127.0.0.1:8000
```

自定义监听地址和端口：

```bash
HOST=0.0.0.0 PORT=9000 ./run_web.sh
```

## 命令行批量分析

对某个图片目录执行批量分析：

```bash
python3 color_analysis.py ./images
```

默认会在图片目录下输出分析结果。

## 开发检查

语法检查：

```bash
python3 -m py_compile color_analysis.py
```

最小人工验证建议：

```text
1. 启动 ./run_web.sh
2. 在单图分析页上传一张图片
3. 确认预览、指标、图表和放大预览正常
4. 在组图分析页上传一个文件夹
5. 确认组图指标、8 张图表和放大预览正常
```

## 说明

- 项目是本地运行工具，不依赖数据库
- 前端为原生 HTML / CSS / JavaScript
- 图表由 `matplotlib` 在后端生成
- 运行时会使用 `.mplconfig/` 作为 Matplotlib 配置目录

## License

如需开源发布，建议后续补充正式许可证文件。
