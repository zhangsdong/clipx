# clipx

[English](./README.md) | 简体中文

## 简介

`clipx` 是一个基于 Python 开发的开源工具库，用于快速、灵活地去除图片背景，支持命令行和 Python 两种调用方式。当前实现集成了 **U2Net** 和 **CascadePSP** 两种图像分割模型，既可单独使用，也可联合使用以提高背景去除效果。此外，`clipx` 的架构设计易于扩展，未来可轻松添加更多模型。

---

## 特性

- 支持单模型或多模型组合调用，灵活控制抠图效果。
- 命令行接口简单易用，适合自动化工作流。
- Python API 简洁易于集成。
- 可单独输出 mask 图像或直接生成去除背景后的透明图像。

---

## 安装

```bash
pip install clipx
```

---

## 命令行使用说明

查看帮助信息：

```bash
clipx --help
```

输出帮助信息如下：

```text
Usage: clipx [OPTIONS]

Options:
  -i, --input FILE/FOLDER   输入图片或文件夹。[必选]
  -o, --output FILE/FOLDER  输出图片或文件夹。
  -m, --model MODEL         使用的模型：u2net, cascadepsp, auto（默认）。
  -k, --only-mask           仅输出 mask 图像。
  -u, --use-mask FILE       使用已有的 mask 图像。
  -v, --version             显示版本信息。
  -h, --help                显示帮助信息。
  --fast                    使用快速模式进行 CascadePSP 优化（速度更快但精度稍低）。
```

---

## 命令行常用示例

- 使用 U2Net 生成 mask 并用 CascadePSP 优化，去除图片背景：

```bash
clipx -i input.jpg -o output.png
```

- 使用 U2Net 去除图片背景：

```bash
clipx -m u2net -i input.jpg -o output.png
```

- 使用 U2Net 只生成 mask 图片：

```bash
clipx -m u2net -i input.jpg -o mask.png -k
```

- 使用 CascadePSP 优化已有的 mask 图片：

```bash
clipx -m cascadepsp -i input.jpg -u mask.png -o output_mask.png
```

- 使用已有的 mask 图片去除背景：

```bash
clipx -i input.jpg -u mask.png -o output.png
```

---

## Python API 使用示例

```python
from clipx import ClipX

# 初始化 ClipX
clip = ClipX(model='auto')

# 使用 auto 模式去除背景
clip.remove_bg('input.jpg', output='output.png')

# 使用 U2Net 仅生成 mask
clip.generate_mask('input.jpg', output='mask.png', model='u2net')
```

---

## 致谢与代码来源

`clipx` 的 CascadePSP 模块基于 [CascadePSP 项目](https://github.com/hkchengrex/CascadePSP) 的 `segmentation_refinement` 实现，根据原项目的许可证条款修改和使用。感谢原作者的工作和贡献。

---

## 开源协议

本项目采用 [MIT 协议](LICENSE) 开源。

