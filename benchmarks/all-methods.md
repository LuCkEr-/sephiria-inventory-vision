# Sephiria inventory method comparison

All methods use inventory-grid icon pixels only. Tooltip and label pixels are excluded.

Challenge capture: **Sephiria_2026-08-07_16-08-37.png**. It is excluded from training/reference data.

| Method | Correct covered items | Covered item accuracy | Identity time / cell |
|---|---:|---:|---:|
| opencv-game-asset-template | 14/14 | 100.0% | 64.426 ms |
| vision-features | 14/14 | 100.0% | 0.212 ms |
| perceptual-hash | 14/14 | 100.0% | 0.017 ms |
| edge-shape | 14/14 | 100.0% | 0.043 ms |
| orb | 14/14 | 100.0% | 0.277 ms |
| hybrid-retrieval | 14/14 | 100.0% | 0.186 ms |
| tiny-cnn-classifier | 0/14 | 0.0% | 3.361 ms |
| cnn-embedding | 11/14 | 78.6% | 1.031 ms |
| siamese-embedding | 14/14 | 100.0% | 0.350 ms |

The dense-window detector localized 0/18 occupied cells and localized plus classified 0/18.

Deep methods are feasibility results on a small corpus. The JSON report contains failures, all-cell scores, training times, and leave-one-screenshot-out classical scores.
