#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CSV="data/nashville_freeway_anomaly.csv"

if [[ ! -f "$CSV" ]]; then
  echo "ERROR: Dataset not found at: $HERE/$CSV"
  echo "Put the CSV at: GTG+TGCN/nashville_freeway_anomaly/data/nashville_freeway_anomaly.csv"
  exit 1
fi

echo "Running GTG+TGCN..."
python3 ft_aed_minimal.py \
  --model gtg_tgcn \
  --csv "$CSV" \
  --lookback 12 \
  --epochs 30 \
  --spatial_lateral 1 \
  --score_agg p90_z

echo
echo "Running GraphSAGE..."
python3 ft_aed_minimal.py \
  --model sage \
  --csv "$CSV" \
  --lookback 12 \
  --epochs 30 \
  --spatial_lateral 1 \
  --score_agg p90_z

echo
echo "Running GDN-style..."
python3 ft_aed_minimal.py \
  --model gdn \
  --csv "$CSV" \
  --lookback 12 \
  --epochs 30 \
  --spatial_lateral 1 \
  --score_agg p90_z \
  --topk 10
