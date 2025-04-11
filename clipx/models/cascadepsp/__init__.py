"""
CascadePSP module for segmentation refinement.
Original project: https://github.com/hkchengrex/CascadePSP
"""

from clipx.models.cascadepsp.model import CascadePSPPredictor
from clipx.models.cascadepsp.download import get_model_path, download_cascadepsp_model

__all__ = ['get_model_path', 'download_cascadepsp_model', 'download_and_or_check_model_file']