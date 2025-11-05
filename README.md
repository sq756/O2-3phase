# O2-3phase

工具用于将灰度图像中两个不同相自动分离，并生成分割掩膜及边界覆盖图。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 命令行使用

```bash
python -m phase_segmentation.cli <输入图像路径>
```

默认会生成两个文件：

- `<原图文件名>_mask.png`：二值掩膜图，分离出目标相；
- `<原图文件名>_overlay.png`：在原图上用红色描绘分界线的覆盖图。

常用参数：

- `--blur-radius`：阈值前的高斯模糊半径（像素），默认 `1.5`；
- `--morphology-radius`：形态学处理的半径（像素），默认 `2`；
- `--prefer-bright-phase`：若目标为亮相，添加此参数即可；
- `--use-opening`：若需要消除小亮点，可改用开运算；
- `--no-overlay` 或 `--no-mask`：只输出其中一种结果；
- `--mask-path` / `--overlay-path`：自定义输出路径。

## Python 调用示例

```python
from phase_segmentation import Segmenter
from PIL import Image

segmenter = Segmenter(blur_radius=1.5, morphology_radius=2, select_dark_phase=True)
with Image.open("input.png") as img:
    result = segmenter.segment(img)

result.save_mask("input_mask.png")
with Image.open("input.png") as img:
    overlay = result.overlay_on(img)
overlay.save("input_overlay.png")
```
