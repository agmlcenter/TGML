GTG+TGCN — Nashville Freeway Anomaly (FT-AED)
============================================

This directory contains a cleaned, reproducible implementation for the Nashville Freeway
Anomaly Detection task used in our paper.

Only these models are included:
  1) GTG+TGCN Autoencoder
  2) GraphSAGE Autoencoder
  3) GDN-style Autoencoder

All other model variants and extra experiments are intentionally removed to keep this
folder minimal and easy to run.


DIRECTORY STRUCTURE
-------------------
GTG+TGCN/nashville_freeway_anomaly/
  - ft_aed_minimal.py        Main training + evaluation script (prints JSON results)
  - requirements.txt         Python dependencies
  - run_examples.sh          Example commands to reproduce results
  - data/                    Place dataset CSV here


DATASET (DOWNLOAD)
------------------
Download the dataset from:
  https://acoursey3.github.io/ft-aed/

Place the CSV file here:
  GTG+TGCN/nashville_freeway_anomaly/data/nashville_freeway_anomaly.csv

Required columns:
  - day, unix_time, milemarker
  - For each lane L in {1..4}:
      lane{L}_speed, lane{L}_volume, lane{L}_occ
  - Labels:
      human_label, crash_record


TIME UNIT NOTE (IMPORTANT)
--------------------------
The dataset is sampled every 30 seconds.

All evaluation/detection windows inside the code are in SECONDS.
The --lookback argument is the number of TIME STEPS (frames), not weeks.

Example:
  --lookback 12  =>  12 * 30 seconds  =  6 minutes of history


ENVIRONMENT SETUP (Debian/Ubuntu)
---------------------------------
From inside this directory:

  cd GTG+TGCN/nashville_freeway_anomaly
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt


TORCH-GEOMETRIC INSTALL NOTE
----------------------------
torch-geometric depends on your installed PyTorch (CPU vs CUDA).
If "pip install -r requirements.txt" fails:
  1) install PyTorch first (CPU or CUDA build)
  2) then install a torch-geometric build compatible with that PyTorch version


RUN EXAMPLES
------------
From inside GTG+TGCN/nashville_freeway_anomaly/

1) GTG+TGCN:
  python3 ft_aed_minimal.py \
    --model gtg_tgcn \
    --csv data/nashville_freeway_anomaly.csv \
    --lookback 12 \
    --epochs 30 \
    --spatial_lateral 1 \
    --score_agg p90_z

2) GraphSAGE:
  python3 ft_aed_minimal.py \
    --model sage \
    --csv data/nashville_freeway_anomaly.csv \
    --lookback 12 \
    --epochs 30 \
    --spatial_lateral 1 \
    --score_agg p90_z

3) GDN-style:
  python3 ft_aed_minimal.py \
    --model gdn \
    --csv data/nashville_freeway_anomaly.csv \
    --lookback 12 \
    --epochs 30 \
    --spatial_lateral 1 \
    --score_agg p90_z \
    --topk 10

Run all:
  ./run_examples.sh


OUTPUT
------
At the end of each run, the script prints a JSON block including:
  - Validation AUC (time-level)
  - Nominal reconstruction error
  - Event-level metrics at multiple target time-FPR points
    (including the 5% row used for reporting)


REPRODUCIBILITY
---------------
To reproduce paper results, use the same:
  - dataset file
  - train/validation day split
  - random seed
  - model hyperparameters
as reported in the paper.
