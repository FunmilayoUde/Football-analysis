#!/bin/bash
set -e

source .venv/bin/activate

VIDEO="${1:-short_sample/HILAL-HAZM_match_B_up7.mp4}"

TS=$(date +%Y%m%d-%H%M%S)
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
DIRTY=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  DIRTY="-dirty"
fi
RUN_ID="${TS}_${SHA}${DIRTY}"
OUT_DIR="outputs_test/${RUN_ID}"

mkdir -p "$OUT_DIR"
echo "[run] video:   $VIDEO"
echo "[run] out_dir: $OUT_DIR"

python3 main.py --source-video "$VIDEO" --out-dir "$OUT_DIR" --enable-team 2>&1 | tee "$OUT_DIR/run.log"

echo "[run] done -> $OUT_DIR"
