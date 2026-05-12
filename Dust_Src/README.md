# Paper Code Export

This directory is a cleaned export of the code used across the thesis/article workflow.
It is organized by pipeline stage so the repository can be uploaded as a paper reference.

## Structure

- `01_data_preparation`
  - Data building and preprocessing scripts used to assemble the modeling table.
- `02_gtg_and_baselines`
  - Main GTG benchmark pipeline and baseline comparison models.
  - `gtg.py` is the central script for graph construction, training, evaluation, and GTG explanations.
  - `other_models.py` reuses `gtg.py` utilities, so both are kept together.
- `03_rl_policy`
  - Reinforcement-learning alert-policy environment, training, and evaluation scripts.
- `04_explanations_and_validation`
  - Explanation mapping, interaction mining, anomaly plots, and statistical validation.
  - `validate_interaction_claims.py` imports `mine_parameter_interactions.py`, so both are kept together.
- `05_paper_reference`
  - Main article/report LaTeX sources for cross-reference with the code.

## Notes

- This export contains scripts only. Large datasets, model artifacts, and generated outputs were intentionally left out.
- The main input table used throughout the modeling pipeline is:
  - `GRID3km_DAILY_20x20_modelready_labeled_featured.csv`
- Some older exploratory scripts in the main workspace were not copied because they are not part of the final reported pipeline.
