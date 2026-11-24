# Crack Analysis Scripts

This folder contains a cleaned, repository-ready version of the supplied analysis scripts.

## Main changes

- Removed year-dependent CLD compensation.
- Removed year-dependent heatmap gain.
- Replaced year-dependent pruning with a fixed pruning setting.
- Removed temporal smoothing and monotonic constraints from physical-length estimation.
- Removed the dilated "damage influence zone" from the growth overlay.
- Changed heatmap normalization to a shared range across compared images.
- Kept control points and check points separate in the 3D similarity transform.
- Removed default pass/fail accuracy thresholds from the transform solver.
- Reduced duplicated and presentation-oriented comments.
- Consolidated duplicated skeleton analysis in `skeleton.py`; `calabdimation.py` is now a compatibility entry point.

## Files

- `data.py`: per-image physical scaling and skeleton-length estimation.
- `cal.py`: comparable distance-transform heatmaps.
- `skeleton.py`: fixed-parameter skeleton and junction analysis.
- `calabdimation.py`: compatibility entry point for `skeleton.py`.
- `growth.py`: registered-mask growth overlay.
- `newgrowth.py`: direct plotting of Table 11 values.
- `solve_facade_transform_table_final.py`: 3D similarity transform and independent check-point residual evaluation.

## Reproducibility note

All thresholds and morphological parameters remain explicit constants in the source code.
They should be justified in the paper or sensitivity-tested if they materially affect the reported results.
