# Usage

## Objectives

- **V2 (target):** `J = −sqrt(Σ_q w_q ((q − q*)/s_q)²)` over `q ∈ {σ_y, n, K, σ_u}`. It is maximal (0) at the target. Here `s_q = |q*|`, and a saturating penalty applies outside the grain-count band. The targets and weights are in `objective.v2` of each config (AZ31: σ_y = 180 MPa, n = 0.30, K = 580 MPa, σ_u = 330 MPa; weights 1, 5, 0.5, 1).
- **V3:** V2 with `s_q` set to the seed-cloud standard deviation, so misses read in seed-σ. Used for the achievability grid.
- **V1 (toughness):** the NeurIPS toughness objective.

Feasibility gates:
- **before simulation:** a decoded image with mean per-channel pixel std below 5 is rejected, as is an RVE with fewer than `codec.min_grains` grains;
- **after simulation:** a DAMASK run that did not converge is rejected.

## Seed caches

`data/seed_caches/seed_cache_<alloy>_<objective>_<decoder>.npz` holds `Z` (the encoder latents of 100 simulated seed RVEs), `Y` (their objective) and `success`. Two sidecar files go with it: `.properties.json` holds the per-seed mechanical properties that warm-start MERIDIAN's property heads, and `.manifest.json` holds the per-seed simulation id and grain count. `data/seed_caches/fig15/AZ31_512/` holds the caches rescored for each cell of the achievability grid.

To rebuild a cache from your own DAMASK seed simulations:

```bash
python scripts/encode_runs_to_seed_cache.py --config configs/acta2026/az31_fmdit_meridian_v1.yaml \
    --runs-dir simulations/runs --rve-root /path/to/rve/AZ31_extruded \
    --encoder-ckpt weights/checkpoints/vitfmdit.pth \
    --output data/seed_caches/seed_cache_AZ31_v1_fmdit.npz --sample 100 --sample-seed 42
python scripts/rescore_seed_cache.py --config configs/acta2026/az31_fmdit_meridian_v2.yaml \
    --seed-cache data/seed_caches/seed_cache_AZ31_v1_fmdit.npz --output data/seed_caches/seed_cache_AZ31_v2_fmdit.npz
```

## Configuration

A run is fully specified by one YAML file. Command-line overrides use `--override key.sub=value`. The main sections are:

| Section | Controls |
|---|---|
| `latent` | dimension, box `bounds`, spatial tokens, seed initialisation |
| `decoder` | `fmdit` (`target_dim` 512–1280), `vitdit`, or `mock`; checkpoint, sampler (steps, noise temperature, shift, guidance) |
| `codec` | class means, alloy class, ODF calibration and texture prior, segmentation (`sam:<pps>:<iou>:<min_area>`), voxel spacing, feasibility gates |
| `simulation` | DAMASK load and phase files, target phase, threads, parallel RVEs, timeout |
| `objective` | `v1` / `v2` / `v3` targets, weights, grain-count penalties |
| `optimizer` | `dante` / `turbo` / `baxus` / `meridian` and their hyperparameters |
| `loop` | rounds, seed cache, resume, masking of failed evaluations, initial anchor |

`${MERIDIAN_ROOT}` (this repository) and `${MSED_ROOT}` (weights, default `weights/`) are expanded in all paths.
