# clipx

English | [简体中文](./README_zh.md)

## Introduction

`clipx` is an open-source Python library designed for quick and flexible image background removal. It supports both command-line interface (CLI) and Python API usage. Currently, it integrates two image segmentation models, **U2Net** and **CascadePSP**, which can be used individually or combined for enhanced performance. Additionally, `clipx` has an extensible architecture, making it easy to incorporate additional models in the future.

---


## Installation

```bash
pip install clipx
```

---


## CLI Examples

- Generate transparent image using combined mode (U2Net + CascadePSP):

```bash
clipx -i input.jpg
```

- Specify output path:

```bash
clipx -i input.jpg -o output.png
```

- Use fast mode for CascadePSP (faster but less accurate):

```bash
clipx -i input.jpg -o output.png --fast
```

---

## Python API Example

```python
from PIL import Image
from clipx import remove

# Testing the API
img = Image.open("tests/leaves-8273504_1920.jpg")
result = remove(img)
result.save("api_test_result.png")

# Get only the mask
mask = remove(img, only_mask=True)
mask.save("mask_only.png")
```

---


## Acknowledgements and Code Sources

This project uses code from the following open source projects:

- U2Net: [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/xuebinqin/U-2-Net)
- CascadePSP: [CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement](https://github.com/hkchengrex/CascadePSP)
- Special thanks to [rembg](https://github.com/danielgatis/rembg) project for inspiration on project structure and implementation approach.

We greatly appreciate the original authors' work and contributions.

---

## License

This project is licensed under the [MIT License](LICENSE).