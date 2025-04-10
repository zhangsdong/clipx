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

- Use U2Net only for mask generation:

```bash
clipx -i input.jpg -o output.png -m u2net
```

- Enable debug logging:

```bash
clipx -i input.jpg --debug
```

---

## Python API Example

```python
from clipx import Clipx

# Initialize Clipx
clipx = Clipx(device='auto')

# Process an image with combined model
result_path, processing_time = clipx.process(
    input_path='input.jpg',
    output_path='output.png',
    model='combined',
    threshold=130,
    fast_mode=False
)

print(f"Output saved to: {result_path}")
print(f"Processing time: {processing_time:.2f} seconds")

# Process with U2Net only
clipx.process(
    input_path='input.jpg',
    output_path='output_u2net.png',
    model='u2net'
)

# Process with CascadePSP only in fast mode
clipx.process(
    input_path='input.jpg',
    output_path='output_cascadepsp.png',
    model='cascadepsp',
    fast_mode=True
)
```

---

## Advanced Usage

### Logging Configuration

You can configure the logging level:

```python
from clipx import set_log_level, enable_console_logging
import logging

# Enable console logging
enable_console_logging()

# Set log level
set_log_level(logging.DEBUG)
```

---

## Acknowledgements and Code Sources

The image segmentation models in `clipx` are based on the following projects:

- U2Net: [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/xuebinqin/U-2-Net)
- CascadePSP: [CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement](https://github.com/hkchengrex/CascadePSP)

We greatly appreciate the original authors' work and contributions.

---

## License

This project is licensed under the [MIT License](LICENSE).