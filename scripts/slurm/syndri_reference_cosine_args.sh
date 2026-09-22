# Source from metric job scripts; optional additional Syndri evaluation case.
SYNDRI_COSINE_ARGS=()
case "${SYNDRI_REFERENCE_COSINE:-0}" in
  1|true|TRUE)
    SYNDRI_COSINE_ARGS=(
      --syndri-reference-cosine
      --syndri-cosine-window "${SYNDRI_COSINE_WINDOW:-5}"
      --syndri-reference-profiles "${SYNDRI_REFERENCE_PROFILES:-/home/jwinchel/Syndri/data/cache/rctd_profiles}"
      --syndri-cell-type-key "${SYNDRI_CELL_TYPE_KEY:-${LABEL_KEY:-}}"
      --syndri-profile-min-umi "${SYNDRI_PROFILE_MIN_UMI:-10}"
      --syndri-profile-max-umi "${SYNDRI_PROFILE_MAX_UMI:-2000}"
      --syndri-profile-cell-min "${SYNDRI_PROFILE_CELL_MIN:-25}"
      --syndri-profile-max-cells "${SYNDRI_PROFILE_MAX_CELLS:-10000}"
      --syndri-profile-cell-type-col "${SYNDRI_PROFILE_CELL_TYPE_COL:-cell_type}"
    )
    if [[ -n "${REFERENCE_DATA:-}" ]]; then
      SYNDRI_COSINE_ARGS+=(--reference-data "$REFERENCE_DATA")
    fi
    ;;
  0|false|FALSE) ;;
  *) echo 'SYNDRI_REFERENCE_COSINE must be 0/1 or false/true.' >&2; exit 1 ;;
esac
