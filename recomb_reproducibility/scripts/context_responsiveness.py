"""Entry point for the self-contained context intervention experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syndri_repro.cli import main

if __name__ == '__main__':
    main(['context'] + sys.argv[1:])
