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
usage: clipx [-h] [-v] [-i INPUT] [-o OUTPUT] [-m {u2net,cascadepsp,combined}]
             [--device {auto,cpu,cuda}] [--threshold THRESHOLD] [--fast]
             [--debug] [--quiet]

Image background removal and mask generation tool

options:
  -h, --help            show this help message and exit
  -v, --version         显示版本信息并退出
  -i INPUT, --input INPUT
                        输入图片路径
  -o OUTPUT, --output OUTPUT
                        输出图片路径（可选，默认为输入文件所在目录下的
                        input_file_remove.png）
  -m {u2net,cascadepsp,combined}, --model {u2net,cascadepsp,combined}
                        使用的模型（默认：combined）
  --device {auto,cpu,cuda}
                        处理所用设备（默认：auto - 如可用则使用GPU）
  --threshold THRESHOLD
                        二值化掩码生成阈值（0-255，默认：130）
  --fast                使用CascadePSP的快速模式（精度较低但速度更快）
  --debug               启用调试日志
  --quiet               仅显示错误输出
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

- 仅使用U2Net进行处理：

```bash
clipx -i input.jpg -o output.png -m u2net
```

- 仅使用CascadePSP进行处理：

```bash
clipx -i input.jpg -o output.png -m cascadepsp
```

- 使用CascadePSP的快速模式（速度更快但精度较低）：

```bash
clipx -i input.jpg -o output.png --fast
```

- 强制使用CPU处理：

```bash
clipx -i input.jpg -o output.png --device cpu
```

- 调整掩码生成的阈值：

```bash
clipx -i input.jpg -o output.png --threshold 150
```

- 启用调试日志：

```bash
clipx -i input.jpg --debug
```

---

## Python API 使用示例

```python
from clipx import Clipx

# 初始化Clipx
clipx = Clipx(device='auto')

# 使用组合模型处理图像
result_path, processing_time = clipx.process(
    input_path='input.jpg',
    output_path='output.png',
    model='combined',
    threshold=130,
    fast_mode=False
)

print(f"输出已保存至: {result_path}")
print(f"处理时间: {processing_time:.2f} 秒")

# 仅使用U2Net处理
clipx.process(
    input_path='input.jpg',
    output_path='output_u2net.png',
    model='u2net'
)

# 仅使用CascadePSP的快速模式处理
clipx.process(
    input_path='input.jpg',
    output_path='output_cascadepsp.png',
    model='cascadepsp',
    fast_mode=True
)
```

---

## 高级用法

### 日志配置

你可以配置日志级别：

```python
from clipx import set_log_level, enable_console_logging
import logging

# 启用控制台日志
enable_console_logging()

# 设置日志级别
set_log_level(logging.DEBUG)
```

---

## 致谢与代码来源

`clipx` 中的图像分割模型基于以下项目：

- U2Net: [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/xuebinqin/U-2-Net)
- CascadePSP: [CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement](https://github.com/hkchengrex/CascadePSP)

感谢这些原始项目作者的工作和贡献。

---

## 开源协议

本项目采用 [MIT 协议](LICENSE) 开源。