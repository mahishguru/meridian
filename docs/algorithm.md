# MERIDIAN

Each round of `suggest()` does the following:

1. **Deep-kernel surrogate** (`dkl.py`). An MLP trunk (`[512, 128] → 16` features) is trained jointly with:
   - a logistic feasibility head;
   - heteroscedastic property heads for `σ_y, σ_u, n, K, N_grains`.

   An exact GP with an ARD Matérn-5/2 kernel is then fit on the learned features of the feasible data.
2. **Search geometry** (`subspace.py`, `optimizer.py`):
   - Per-axis anisotropy weights come from the active subspace of the surrogate-gradient covariance. For the first rounds they come from PCA of a class latent pool, when one is provided.
   - A Sobol cloud of candidates is drawn around the incumbent within an anisotropic trust region, with sparse coordinate perturbations and a fraction of diffuse scout moves.
   - Candidates are optionally projected onto the adaptive norm shell of the feasible latents.
3. **Acquisition** (`batch.py`, `property_acquisition.py`): feasibility-gated qLogNEI, blended with a Monte-Carlo expected improvement of the target objective computed through the property heads. The property-head term includes a saturating grain-count penalty.
4. **Batch selection**: greedy determinantal-point-process selection over the top candidates, or a portfolio batch (exploit, diffuse scout, UCB scout, DPP fill). Optional slots hold a property-gradient seed (`gradient_seed.py`) and the incumbent.
5. **Trust-region cadence and restarts** (`trust_region.py`, `mcts_restart.py`): a TuRBO-style success/failure cadence with a capped failure tolerance. Collapse or a plateau triggers a restart from a small surrogate-side MCTS or from the class pool. After a set round, a final polishing phase takes over.

All hyperparameters are in the `optimizer.meridian` block of each config. The published settings are those in `configs/acta2026/` and `configs/neurips2026/`.

## Using MERIDIAN on your own black-box function

MERIDIAN depends only on PyTorch, BoTorch and GPyTorch, not on the decoder or DAMASK. It maximises its objective; return `NaN` or `-inf` for failed evaluations.

```python
from meridian.config import _to_node
from meridian.optimizers.meridian import MeridianOptimizer
import yaml

cfg = _to_node(yaml.safe_load(open("configs/base.yaml"))["optimizer"]["meridian"])
cfg.update(adaptive_shell=False, shell_radius_min=0.0, shell_radius_max=0.0)  # shell = encoder-latent specific

opt = MeridianOptimizer(dim=20, bounds=(-3.0, 3.0), batch_size=4, meridian_cfg=cfg, device="cpu")
opt.update(X_init, y_init)              # initial design (required before the first suggest)
for _ in range(20):
    X = opt.suggest()                   # (4, 20)
    opt.update(X, f(X))
```

A complete, runnable version on a 20-D Ackley function with an infeasible region is in [`examples/standalone_blackbox.py`](../examples/standalone_blackbox.py).
