"""
Bearing Fault Detection - Deep Learning System

A production-ready machine learning system for automated bearing fault diagnosis
using multi-scale attention fusion and advanced signal processing.
"""

__version__ = "1.0.0"
__author__ = "Shawon"

from . import models
from . import datasets
from . import training
from . import utils

__all__ = ['models', 'datasets', 'training', 'utils']
