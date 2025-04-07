# clipx

English | [简体中文](./README_zh.md)

## Introduction

`clipx` is an open-source Python library designed for quick and flexible image background removal. It supports both command-line interface (CLI) and Python API usage. Currently, it integrates two image segmentation models, **U2Net** and **CascadePSP**, which can be used individually or combined for enhanced performance. Additionally, `clipx` has an extensible architecture, making it easy to incorporate additional models in the future.

---

## Features

- Supports single or combined model usage for flexible background removal.
- User-friendly CLI suitable for automation workflows.
- Simple and easy-to-integrate Python API.
- Can output mask images separately or directly produce transparent background images.

---

## Installation

```bash
pip install clipx
```

---

## CLI Usage

View help information:

```bash
clipx --help
```

Output help information:

```text
Usage: clipx [OPTIONS]

Options:
  -i, --input FILE/FOLDER   Input image or folder. [required]
  -o, --output FILE/FOLDER  Output image or folder.
  -m, --model MODEL         Model to use: u2net, cascadepsp, auto (default).
  -k, --only-mask           Output only mask image.
  -u, --use-mask FILE       Use an existing mask image.
  -v, --version             Show version information.
  -h, --help                Show this help message.
  --fast                    Use fast mode for CascadePSP (less accurate but faster).
```

---

## CLI Examples

- Generate mask with U2Net and refine with CascadePSP to remove background:

```bash
clipx -i input.jpg -o output.png
```

- Remove background using U2Net:

```bash
clipx -m u2net -i input.jpg -o output.png
```

- Generate mask only with U2Net:

```bash
clipx -m u2net -i input.jpg -o mask.png -k
```

- Refine an existing mask with CascadePSP:

```bash
clipx -m cascadepsp -i input.jpg -u mask.png -o output_mask.png
```

- Remove background using an existing mask image:

```bash
clipx -i input.jpg -u mask.png -o output.png
```

---

## Python API Example

```python
from clipx import ClipX

# Initialize ClipX
clip = ClipX(model='auto')

# Remove background using auto mode
clip.remove_bg('input.jpg', output='output.png')

# Generate mask using U2Net only
clip.generate_mask('input.jpg', output='mask.png', model='u2net')
```

---

## Acknowledgements and Code Sources

The CascadePSP module in `clipx` is based on the `segmentation_refinement` implementation from the [CascadePSP project](https://github.com/hkchengrex/CascadePSP), modified and used according to the original project's licensing terms. We greatly appreciate the original author's work and contributions.

---

## License

This project is licensed under the [MIT License](LICENSE).

