# clipx

[English](./README.md) | 简体中文

## 简介

`clipx` 是一个基于 Python 开发的开源工具库，用于快速、灵活地去除图片背景，支持命令行和 Python 两种调用方式。当前实现集成了 **U2Net** 和 **CascadePSP** 两种图像分割模型，既可单独使用，也可联合使用以提高背景去除效果。此外，`clipx` 的架构设计易于扩展，未来可轻松添加更多模型。


---

## 安装

```bash
pip install clipx
```

---


## 命令行常用示例

- 使用组合模式（U2Net + CascadePSP）生成透明图像：

```bash
clipx -i input.jpg
```

- 指定输出路径：

```bash
clipx -i input.jpg -o output.png
```

- 使用CascadePSP的快速模式（速度更快但精度较低）：

```bash
clipx -i input.jpg -o output.png --fast
```

---

## Python API 使用示例

```python
from PIL import Image
from clipx import remove

# 测试 API
img = Image.open("tests/leaves-8273504_1920.jpg")
result = remove(img)
result.save("api_test_result.png")

# 只获取掩码
mask = remove(img, only_mask=True)
mask.save("mask_only.png")
```

---


## 致谢与代码来源

本项目使用了以下开源项目的代码：

- U2Net: [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/xuebinqin/U-2-Net)
- CascadePSP: [CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement](https://github.com/hkchengrex/CascadePSP)
- 特别感谢 [rembg](https://github.com/danielgatis/rembg) 项目对项目结构和实现方法的启发。

感谢这些原始项目作者的工作和贡献。

---

## 开源协议

本项目采用 [MIT 协议](LICENSE) 开源。