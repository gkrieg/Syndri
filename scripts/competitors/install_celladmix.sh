#!/usr/bin/env bash
set -euo pipefail
lib=/home/jwinchel/.local/lib/R/competitors
r=/home/jwinchel/miniconda3/envs/r_env/bin/R
rscript=/home/jwinchel/miniconda3/envs/r_env/bin/Rscript
export R_LIBS_USER="$lib"
export R_MAKEVARS_USER=/home/jwinchel/Syndri/scripts/competitors/R.Makevars
mkdir -p "$lib"
if ! "$rscript" -e 'quit(status=if (requireNamespace("nnls", quietly=TRUE)) 0 else 1)' >/dev/null 2>&1; then
  "$rscript" -e 'install.packages("nnls", repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))'
fi
if ! "$rscript" -e 'quit(status=if (requireNamespace("CRF", quietly=TRUE)) 0 else 1)' >/dev/null 2>&1; then
  scratch=$(mktemp -d /tmp/syndri-crf.XXXXXXXX)
  trap 'rm -rf "$scratch"' EXIT
  curl -fsSL https://cran.r-project.org/src/contrib/Archive/CRF/CRF_0.4-3.tar.gz -o "$scratch/CRF.tar.gz"
  echo '6e476f83a026ee28403f82eb85c26515c65c6ede24aeafc3d7e78be65c608c4a  '"$scratch/CRF.tar.gz" | sha256sum -c -
  tar -xzf "$scratch/CRF.tar.gz" -C "$scratch"
  sed -i 's/\bCalloc(/R_Calloc(/g; s/\bFree(/R_Free(/g' "$scratch/CRF/src/misc.h"
  "$r" CMD INSTALL -l "$lib" "$scratch/CRF"
fi
if ! "$rscript" -e 'quit(status=if (requireNamespace("cellAdmix", quietly=TRUE)) 0 else 1)' >/dev/null 2>&1; then
  "$r" CMD INSTALL -l "$lib" /home/jwinchel/cellAdmix
fi
