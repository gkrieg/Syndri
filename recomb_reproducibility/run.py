#!/usr/bin/env python
"""Run from this directory or any other working directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syndri_repro.cli import main

if __name__ == '__main__':
    main()

