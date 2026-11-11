"""Syndri-only reproduction of RECOMB 2027 submission 395."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/.matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'outputs/.numba'))
__version__ = '0.1.0'

