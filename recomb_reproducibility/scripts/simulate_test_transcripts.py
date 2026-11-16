#!/usr/bin/env python3
"""Sample coordinates for a held-out simulated CosMx test slice."""
import os
import sys
from pathlib import Path

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(variable, '8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syndri_repro.transcript_locations import main

if __name__ == '__main__':
    main()
