# Released Results Bundle

This directory contains lightweight result artifacts referenced by the paper and one ablation bundle.

## benchmark_and_rl
- `compare_s11_summary.json`: baseline-model comparison summary used for the benchmark table.
- `rl_chosen_threshold_s77.json`: selected PPO alert threshold and validation-time operating point.

## explanations
- `gtg_s77_explanations.json`: GTG positive-event explanations for the seed-77 model.
- `gtg_rl_explanations.json`: PPO-stage explanations exported from the RL pipeline.

## interaction_validation
- `interaction_summary_seed77.json`: summary of mined feature-lag interactions for the main seed.
- `interaction_pairs_shortlist_seed77.csv`: shortlisted recurrent feature-lag pairs.
- `feature_lag_stats_seed77.csv`: per-feature lag and anomaly statistics.
- `validation_summary_multi_seed.json`: stability / permutation / temporal-holdout summary across seeds 11, 42, 77.
- `seed_stability_pairwise_jaccard.csv`: overlap of top interaction pairs across seeds.
- `permutation_significance.csv`: permutation-test significance for top interaction pairs.
- `temporal_holdout_confirmation.csv`: forward holdout confirmation counts for top pairs.

## 07_ablation
- `GEO_train2.py`: ablation training script.
- `results/ablation_results_daily_v6.json`: compact ablation summary.
- `results/ablation_results.zip`: per-run metrics JSON files and saved model checkpoints for the ablation sweep.
