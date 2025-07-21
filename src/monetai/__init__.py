"""
MonetAI: CycleGAN for Photo to Monet Style Transfer

A deep learning project implementing CycleGAN for transforming photographs
into Monet-style paintings using unpaired image-to-image translation.
"""

__version__ = "0.1.0"
__author__ = "Alankrit Singh"
__email__ = "f20221186@goa.bits-pilani.ac.in"

from .models import CycleGAN
from .data import DataLoader
from .utils import *

__all__ = ["CycleGAN", "DataLoader"]
