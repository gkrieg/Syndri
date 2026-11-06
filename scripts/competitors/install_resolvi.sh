#!/usr/bin/env bash
set -euo pipefail
env_dir=/home/jwinchel/.venv/resolvi
if [[ ! -x "$env_dir/bin/python" ]]; then
  /home/jwinchel/miniconda3/envs/rctd/bin/python -m venv --system-site-packages "$env_dir"
fi
if ! "$env_dir/bin/python" -c 'import scvi; assert scvi.__version__ == "1.3.3"' >/dev/null 2>&1; then
  "$env_dir/bin/python" -m pip install 'scvi-tools==1.3.3'
fi
