"""
This module contains code modified from the CascadePSP project
The implementation is based on the segmentation-refinement code
Original code: https://github.com/hkchengrex/CascadePSP
"""

import cv2
import time
import matplotlib.pyplot as plt
import cascadepsp as refine
image = cv2.imread('test/aeroplane.jpg')
mask = cv2.imread('test/aeroplane.png', cv2.IMREAD_GRAYSCALE)

# model_path can also be specified here
# This step takes some time to load the model
refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
mask_output = refiner.refine(image, mask, fast=False, L=900)

plt.imshow(mask_output)
plt.show()