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

- 仅使用U2Net进行处理：

```bash
clipx -i input.jpg -o output.png -m u2net
```


- 启用调试日志：

```bash
clipx -i input.jpg --debug
```

---

## Python API 使用示例

```python
from clipx import remove_background

result = remove_background("photo.jpg")

print(f"Image with background removed saved to: {result}")
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