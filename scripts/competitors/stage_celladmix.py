"""Convert formatted transcripts into cellAdmix CSV, preserving source files."""
from __future__ import annotations
import argparse
import os
import tempfile
from pathlib import Path
import pyarrow.csv as csv
import pyarrow.parquet as pq
from common import transcript_path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('final_h5ad', type=Path)
    p.add_argument('output_csv', type=Path)
    p.add_argument('--transcript-root', type=Path)
    args = p.parse_args()
    if args.output_csv.exists():
        raise FileExistsError(args.output_csv)
    table = pq.read_table(transcript_path(args.final_h5ad, args.transcript_root))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.celladmix-', suffix='.csv', dir=args.output_csv.parent)
    os.close(fd)
    try:
        with open(tmp, 'wb') as handle:
            csv.write_csv(table, handle)
        os.link(tmp, args.output_csv)
    finally:
        os.unlink(tmp)
