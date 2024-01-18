"""
This module is intended to ease the use of specific tasks such as data preprocessing or model training. 
"""

from .preprocess import PreprocessPipeline
from .training import TrainingPipeline

__all__ = ["PreprocessPipeline", "TrainingPipeline"]
