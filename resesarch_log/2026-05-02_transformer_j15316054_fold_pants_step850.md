# Experiment: transformer_j15316054 — fold_pants — step 850

- **Date**: 2026-05-02
- **Task**: `fold_pants`
- **Model**: transformer
- **Checkpoint**: `exp/2026-05-02_07-12-25_transformer_j15316054/checkpoints/step000850.pt`
- **SLURM Job**: 15316054
- **Data**: `/iris/u/abhijnya/droid-robot/demos/test`, `/iris/u/am208/droid-robot/demos/test`, `/iris/u/am208/droid-robot/preferences`
- **Cross-preferences**: `/iris/u/abhijnya/droid-robot/cross_preferences`, `/iris/u/am208/droid-robot/cross_preferences`

## Dataset

- 136 scored trajectories (paired + standalone)
- 11 preference dimensions:
  Quality of 1st fold, Wrinkle of 1st fold,
  Quality of 2nd fold, Wrinkle of 2nd fold,
  Quality of 3rd fold, Wrinkle of 3rd fold,
  Alignment of final fold, Fast, Smooth,
  Damage to environment, Overall quality

## Score Statistics

| Dimension              |   min   |   max   |  mean  |  std  |  p10    |  p50   |  p90   |
|------------------------|---------|---------|--------|-------|---------|--------|--------|
| Quality of 1st fold    | -17.040 |  16.554 |  0.081 | 8.234 | -11.337 |  0.546 | 11.108 |
| Wrinkle of 1st fold    | -18.593 |  15.591 | -1.305 | 8.304 | -12.954 | -0.783 | 10.046 |
| Quality of 2nd fold    | -19.446 |  15.800 | -2.882 | 8.976 | -14.986 | -3.337 |  9.198 |
| Wrinkle of 2nd fold    | -16.523 |  14.572 | -0.648 | 7.891 | -11.405 | -1.032 | 10.495 |
| Quality of 3rd fold    | -17.838 |  14.695 | -1.370 | 7.265 | -12.513 | -0.802 |  7.437 |
| Wrinkle of 3rd fold    | -16.383 |  13.283 | -1.759 | 6.645 | -12.217 | -1.338 |  6.759 |
| Alignment of final fold| -15.820 |  15.402 |  1.961 | 8.145 | -10.373 |  1.876 | 12.020 |
| Fast                   | -15.348 |  18.359 |  2.643 | 9.064 | -10.063 |  3.045 | 14.962 |
| Smooth                 | -13.381 |  14.862 |  0.575 | 5.938 |  -7.791 |  1.138 |  7.775 |
| Damage to environment  | -10.402 |  13.000 |  2.388 | 6.196 |  -6.106 |  2.739 | 10.441 |
| Overall quality        | -20.374 |  14.959 | -3.634 | 8.398 | -14.051 | -4.277 |  8.066 |

## Pairwise Correlations

```
                      Q1st   W1st   Q2nd   W2nd   Q3rd   W3rd   Align  Fast   Smooth Damage Overall
Q1st fold            1.00   0.96   0.78   0.72   0.67   0.64   0.65   0.77   0.65   0.35   0.82
W1st fold            0.96   1.00   0.78   0.74   0.72   0.70   0.60   0.66   0.49   0.28   0.83
Q2nd fold            0.78   0.78   1.00   0.97   0.73   0.72   0.63   0.67   0.64   0.39   0.93
W2nd fold            0.72   0.74   0.97   1.00   0.74   0.74   0.65   0.65   0.62   0.45   0.91
Q3rd fold            0.67   0.72   0.73   0.74   1.00   0.99   0.81   0.55   0.43   0.35   0.86
W3rd fold            0.64   0.70   0.72   0.74   0.99   1.00   0.80   0.53   0.39   0.38   0.84
Align final          0.65   0.60   0.63   0.65   0.81   0.80   1.00   0.68   0.64   0.70   0.75
Fast                 0.77   0.66   0.67   0.65   0.55   0.53   0.68   1.00   0.68   0.47   0.68
Smooth               0.65   0.49   0.64   0.62   0.43   0.39   0.64   0.68   1.00   0.53   0.64
Damage               0.35   0.28   0.39   0.45   0.35   0.38   0.70   0.47   0.53   1.00   0.44
Overall              0.82   0.83   0.93   0.91   0.86   0.84   0.75   0.68   0.64   0.44   1.00
```

## Key Observations

1. **Quality/Wrinkle pairs collapsed within each fold**: Q↔W correlations are 0.96 (1st), 0.97 (2nd), 0.99 (3rd). The model treats quality and wrinkle as the same signal per fold.

2. **Overall quality is a near-average of fold dims** (r=0.82-0.93), not learning anything independent.

3. **"Damage to environment" is the most independent dim** (r=0.28-0.53 with most others), suggesting the model has captured something distinct here. Strongest coupling is with Alignment (r=0.70).

4. **Fast correlates more with early folds** (r=0.77 with 1st fold) than late folds (r=0.55 with 3rd fold) — possibly speed affects early fold quality more.

5. **Smooth has moderate independence** from fold dims (r=0.39-0.65), weakest correlation with 3rd fold wrinkle (r=0.39).

6. **All correlations are positive** — no dimension is anti-correlated, unlike the bowl task where target-specific dims showed r=-0.57. This suggests the model is mostly learning a single "good trajectory vs bad trajectory" axis.

## Diagnosis

With 136 trajectories and 11 dims, the model is collapsing most dimensions into a general quality signal. The Quality/Wrinkle pairs within each fold are redundant. Consider:
- Merging Quality+Wrinkle per fold (reducing from 11 to 8 dims)
- Adding more cross-preferences to force separation between semantically different dims
- Training longer or with more data to allow fine-grained distinctions to emerge

## Scatter Matrix

![Scatter Matrix](../exp/2026-05-02_07-12-25_transformer_j15316054/checkpoints/vis_step000850/reward_model_2026-05-02_07-12-25_transformer_j15316054_step000850_data_scatter_matrix.png)

## Dimension Histograms

![Dim Histograms](../exp/2026-05-02_07-12-25_transformer_j15316054/checkpoints/vis_step000850/reward_model_2026-05-02_07-12-25_transformer_j15316054_step000850_data_dim_histograms.png)

## Artifacts

- NPZ: `exp/2026-05-02_07-12-25_transformer_j15316054/checkpoints/vis_step000850/reward_model_..._data.npz`
