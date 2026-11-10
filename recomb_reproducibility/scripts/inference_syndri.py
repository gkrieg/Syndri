#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syndri_repro.cli import main
if __name__ == '__main__':
    main(['infer', *sys.argv[1:]])
