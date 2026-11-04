#!/usr/bin/env bash
# Runs every simulation for the SimplyAL revision, then fills the manuscript.
#
# From the SimplyAL repository root, with the v1.1 patch applied and
# simulate_al.py and fill_results.py copied into backend/:
#   bash run_all.sh /abs/path/labels.csv /abs/path/to/paper [label_column] [path_column]
# Rerunning the same command resumes: finished runs are skipped.
# SEEDS=0,1 bash run_all.sh ...   uses two seeds instead of three (faster).
set -euo pipefail
if [ $# -lt 2 ]; then
  sed -n '2,9p' "$0"
  exit 1
fi
CSV="$1"; PAPER="$2"; LABEL="${3:-annotation}"; PATHCOL="${4:-file_path}"
SEEDS="${SEEDS:-0,1,2}"; PY="${PY:-python}"
SIM=backend/simulate_al.py; FILL=backend/fill_results.py
MAIN=runs/main; CPU=runs/cpu
for f in "$SIM" "$FILL" "$CSV" "$PAPER/SimplyAL_revised.tex"; do
  [ -e "$f" ] || { echo "Missing: $f"; exit 1; }
done
mkdir -p runs
exec > >(tee -a runs/run_all.log) 2>&1

data=(--csv "$CSV" --label-column "$LABEL" --path-column "$PATHCOL")
gpu=("${data[@]}" --seeds "$SEEDS" --out "$MAIN")
all3=(--models resnet18,resnet50,dinov2)
stage() { echo; echo "=== $(date '+%F %T')  $*"; }

stage "1/7  Least confidence and random sampling (Figure 5, plateau, latency)"
"$PY" "$SIM" "${gpu[@]}" "${all3[@]}" --strategies least_confidence,random
stage "2/7  Margin, entropy and core-set sampling"
"$PY" "$SIM" "${gpu[@]}" "${all3[@]}" --strategies margin,entropy,diversity
stage "3/7  Learning rate 1e-4"
"$PY" "$SIM" "${gpu[@]}" "${all3[@]}" --strategies least_confidence --lr 1e-4
stage "4/7  Learning rate 1e-5"
"$PY" "$SIM" "${gpu[@]}" "${all3[@]}" --strategies least_confidence --lr 1e-5
stage "5/7  8 and 32 images per episode (ResNet-50)"
for E in 8 32; do
  "$PY" "$SIM" "${gpu[@]}" --models resnet50 --strategies least_confidence --episode-size "$E"
done
stage "6/7  3 and 10 epochs per episode (ResNet-50)"
for P in 3 10; do
  "$PY" "$SIM" "${gpu[@]}" --models resnet50 --strategies least_confidence --epochs "$P"
done
stage "7/7  CPU latency (ResNet-18, one seed, first episodes only)"
"$PY" "$SIM" "${data[@]}" --seeds 0 --models resnet18 --strategies least_confidence \
  --device cpu --max-frac 0.25 --out "$CPU"
stage "Filling the manuscript"
"$PY" "$FILL" --runs "$MAIN" --cpu-runs "$CPU" "${data[@]}" --tex-dir "$PAPER" --repo .
stage "Done. Read $PAPER/results_report.txt next."
