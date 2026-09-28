"""Smoke tests: dry-run the full loop with the mock decoder / oracle for every optimizer.

Run with: `python -m pytest tests/` or `python tests/test_smoke.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


OPTIMIZER_OVERRIDES = {
    "dante": ["optimizer.dante.surrogate_epochs=10", "optimizer.dante.n_rollouts=20",
              "optimizer.dante.n_leaves_per_expand=8"],
    "turbo": ["optimizer.turbo.n_candidates=256"],
    "baxus": [],
    "meridian": ["optimizer.meridian.surrogate_epochs=10", "optimizer.meridian.sobol_cloud=256",
                 "optimizer.meridian.dpp_pool=32"],
}


@pytest.mark.parametrize("optimizer", sorted(OPTIMIZER_OVERRIDES))
def test_dry_run(optimizer, tmp_path):
    """Full loop (decode -> simulate -> score -> update) with mock decoder and oracle."""
    from meridian.config import apply_overrides, load_config
    from meridian.loop import OptimizationLoop

    cfg = load_config(ROOT / "configs" / "base.yaml")
    apply_overrides(cfg, [
        "loop.dry_run=true",
        "loop.n_iterations=2",
        "loop.checkpoint_every=1",
        "latent.init.mode=sobol",
        "latent.init.n_seed=8",
        "optimizer.batch_size=4",
        "decoder.name=mock",
        f"optimizer.name={optimizer}",
        "objective.name=v1",
        "experiment.device=cpu",
        f"experiment.output_dir={tmp_path}",
        f"experiment.name=smoke_{optimizer}",
    ] + OPTIMIZER_OVERRIDES[optimizer])
    res = OptimizationLoop(cfg).run()
    assert "best_objective" in res
    assert len(res["best_z"]) == int(cfg.latent.dim)


def test_calibrated_pipeline_publishes_repainted_image(tmp_path, monkeypatch):
    import meridian.simulation as simulation

    raw_png = tmp_path / "images" / "iter0001_b00.png"
    raw_png.parent.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(raw_png)

    def fake_recon_to_dream3d(recon_png, **kwargs):
        recon_png = Path(recon_png)
        calibrated = recon_png.with_name(f"{recon_png.stem}_calibPW_sam.png")
        Image.fromarray(np.full((8, 8, 3), 173, dtype=np.uint8)).save(calibrated)
        Path(kwargs["output_dream3d"]).touch()
        return Path(kwargs["output_dream3d"]), 300

    class FailedDAMASK:
        def run(self, *_args, **_kwargs):
            return type("Result", (), {"returncode": 1, "hdf5_path": None})()

    monkeypatch.setattr(simulation, "recon_to_dream3d", fake_recon_to_dream3d)
    pipeline = simulation.SimulationPipeline.__new__(simulation.SimulationPipeline)
    pipeline.damask = FailedDAMASK()
    pipeline.class_means_json = tmp_path / "class_means.json"
    pipeline.class_key = "AZ31_extruded"
    pipeline.template_dream3d = None
    pipeline.colour_tol = None
    pipeline.spacing = None
    pipeline.cleanup_hdf5 = False
    pipeline.min_grains = 250
    pipeline.target_size = 10
    pipeline.min_pixel_std = None
    pipeline.calibrate = True
    pipeline.texture_prior = type(
        "TexturePrior", (), {"predict_hist": lambda self, _z: np.ones(4)}
    )()

    outcome = pipeline.evaluate_png(raw_png, tmp_path / "sim", z=np.zeros(4))

    assert not outcome.success
    assert np.asarray(Image.open(raw_png)).mean() == 173
    assert (tmp_path / "sim" / "iter0001_b00__r10_calibPW_sam.png").is_file()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
