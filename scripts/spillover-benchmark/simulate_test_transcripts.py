#!/usr/bin/env python3
"""Sample transcript coordinates from existing held-out SYNDRI-SIM slice counts."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(variable, '8')
sys.path.insert(0, str(ROOT/'recomb_reproducibility'))
sys.path.insert(0, str(ROOT/'Heimdall'))
from syndri_repro.transcript_locations import main

if __name__ == '__main__':
    main()
