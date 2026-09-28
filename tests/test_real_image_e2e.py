#!/usr/bin/env python
"""End-to-end test with a REAL encoded microstructure image.

Uses AZ31_extruded_17_preview.png (300×300 8-bit PNG) from the orientation_codec
repo, runs the full pipeline:

  PNG → orientation_codec → Dream3D → DAMASK CPFEM → stress–strain → properties
  → V1 (toughness) objective
  → V2 (target-driven) objective

Saves all intermediate and final values to a JSON file for inspection.

Usage:
    python tests/test_real_image_e2e.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODEC_DIR = ROOT.parent / "orientation_codec"
ASSETS = ROOT / "assets"
AZ31_DIR = ROOT.parent / "automated_simulations"

IMAGE_PATH = CODEC_DIR / "AZ31_extruded_17_preview.png"


def write_test_load_yaml(path: Path) -> None:
    """Write a load.yaml with reduced increments for faster testing.

    Full production uses t=250, N=1250.
    Here we use t=50, N=250 (5× faster, still enough for property extraction).
    """
    path.write_text("""\
solver:
  mechanical: spectral_basic

loadstep:
  - boundary_conditions:
      mechanical:
        dot_F: [[1.0e-3, 0, 0],
                [0,      x, 0],
                [0,      0, x]]
        P: [[x, x, x],
            [x, 0, x],
            [x, x, 0]]
    discretization:
      t: 50
      N: 250
    f_out: 5
""")


def main() -> int:
    if not IMAGE_PATH.is_file():
        print(f"ERROR: Test image not found: {IMAGE_PATH}")
        return 1

    work = Path(tempfile.mkdtemp(prefix="meridian_real_e2e_"))
    print(f"Working directory: {work}")
    results = {"image": str(IMAGE_PATH), "work_dir": str(work)}

    # ----------------------------------------------------------------
    # Step 1: PNG → Dream3D
    # ----------------------------------------------------------------
    from meridian.simulation.codec import png_to_dream3d

    class_means_json = ASSETS / "class_means.json"
    assert class_means_json.is_file(), f"Missing: {class_means_json}"

    dream3d_path = work / "AZ31_extruded_17.dream3d"
    t0 = time.time()
    png_to_dream3d(
        png_path=IMAGE_PATH,
        class_means_json=class_means_json,
        class_key="AZ31_extruded",
        output_dream3d=dream3d_path,
        spacing=np.array([1.0, 1.0, 1.0]),
    )
    codec_time = time.time() - t0
    results["codec_seconds"] = round(codec_time, 2)

    import h5py
    with h5py.File(dream3d_path, "r") as f:
        base = "DataContainers/SyntheticVolumeDataContainer"
        fids = f[f"{base}/CellData/FeatureIds"][:]
        n_grains = f[f"{base}/Grain Data/EulerAngles"].shape[0]
        grid_shape = list(fids.shape)
    results["grid_shape"] = grid_shape
    results["n_grains"] = int(n_grains)
    print(f"[1/5] Dream3D: grid={grid_shape}, grains={n_grains}  ({codec_time:.1f}s)")

    # ----------------------------------------------------------------
    # Step 2: Dream3D → DAMASK inputs
    # ----------------------------------------------------------------
    from meridian.simulation.damask import DAMASKRunner

    fast_load = work / "load.yaml"
    write_test_load_yaml(fast_load)
    phase_yaml = AZ31_DIR / "AZ31_Phenopower.yaml"
    assert phase_yaml.is_file(), f"Missing: {phase_yaml}"

    runner = DAMASKRunner(
        load_yaml=fast_load,
        phase_yaml=phase_yaml,
        target_phase="AZ31",
        target_homog="SX",
        n_threads=8,
        timeout_seconds=3600,
    )

    sim_dir = work / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    material_file, geom_file = runner._convert_dream3d(dream3d_path, sim_dir)
    print(f"[2/5] DAMASK inputs: {material_file.name}, {geom_file.name}")

    # ----------------------------------------------------------------
    # Step 3: Run DAMASK_grid
    # ----------------------------------------------------------------
    print("[3/5] Running DAMASK_grid (300×300, 250 increments) …")
    t0 = time.time()
    result = runner.run(dream3d_path, sim_dir=sim_dir)
    damask_time = time.time() - t0
    results["damask_seconds"] = round(damask_time, 1)
    results["damask_returncode"] = result.returncode
    print(f"       Return code: {result.returncode}  ({damask_time:.1f}s)")

    if result.returncode != 0 or result.hdf5_path is None:
        print("DAMASK FAILED. Last 40 lines of log:")
        lines = result.log_path.read_text().splitlines()
        for line in lines[-40:]:
            print(f"  | {line}")
        results["error"] = "damask_failed"
        _save_results(results, work)
        return 1

    results["hdf5_path"] = str(result.hdf5_path)

    # ----------------------------------------------------------------
    # Step 4: Post-process → stress–strain → properties
    # ----------------------------------------------------------------
    from meridian.simulation.postproc import StressStrainProcessor
    from meridian.simulation.extractor import PropertyExtractor

    post = StressStrainProcessor(loading_direction="x")
    csv_path = work / "stress_strain.csv"
    post.process_to_csv(result.hdf5_path, csv_path)

    import pandas as pd
    df = pd.read_csv(csv_path)
    results["stress_strain_rows"] = len(df)
    results["max_strain"] = round(float(df["strain_eng"].max()), 6)
    results["max_stress_MPa"] = round(float(df["stress_eng"].max()), 2)
    print(f"[4/5] Stress–strain: {len(df)} points, max strain={results['max_strain']:.4f}, max stress={results['max_stress_MPa']:.1f} MPa")

    ext = PropertyExtractor(youngs_strain_max=1e-3, yield_tolerance=0.4)
    props = ext.extract(csv_path)

    if props is None:
        print("WARNING: Property extraction returned None")
        results["properties"] = None
        results["error"] = "extraction_failed"
        _save_results(results, work)
        return 1

    props_dict = props.to_dict()
    results["properties"] = {k: round(v, 4) if abs(v) < 1e6 else v for k, v in props_dict.items()}
    print(f"       E       = {props.E:.1f} MPa")
    print(f"       σ_y     = {props.sigma_y:.2f} MPa")
    print(f"       σ_u     = {props.sigma_u:.2f} MPa")
    print(f"       ε_uniform = {props.epsilon_uniform:.4f}")
    print(f"       K       = {props.K:.2f} MPa")
    print(f"       n       = {props.n:.4f}")
    print(f"       ε_frac  = {props.epsilon_fracture:.4f}")
    print(f"       σ_frac  = {props.sigma_fracture:.2f} MPa")

    # ----------------------------------------------------------------
    # Step 5: Evaluate both objective functions
    # ----------------------------------------------------------------
    from meridian.objectives.v1_toughness import ToughnessObjective
    from meridian.objectives.v2_target import TargetDrivenObjective

    # V1: toughness-weighted (default config values)
    obj_v1 = ToughnessObjective(
        sigma_y_ref=250.0,
        toughness_ref=50.0,
        alpha=1.0,
        beta=1.0,
        n_min=0.10,
        lambda_n=5.0,
    )
    score_v1 = obj_v1(props)

    # V2R: target-driven (default config target)
    obj_v2 = TargetDrivenObjective(
        targets={"sigma_y": 280.0, "n": 0.22, "K": 480.0, "sigma_u": 380.0},
        weights={"sigma_y": 1.0, "n": 50.0, "K": 0.5, "sigma_u": 1.0},
    )
    score_v2 = obj_v2(props)

    results["objective_v1"] = round(score_v1, 6)
    results["objective_v2"] = round(score_v2, 6)
    results["objective_v1_config"] = {
        "sigma_y_ref": 250.0, "toughness_ref": 50.0,
        "alpha": 1.0, "beta": 1.0, "n_min": 0.10, "lambda_n": 5.0,
    }
    results["objective_v2_config"] = {
        "targets": {"sigma_y": 280.0, "n": 0.22, "K": 480.0, "sigma_u": 380.0},
        "weights": {"sigma_y": 1.0, "n": 50.0, "K": 0.5, "sigma_u": 1.0},
    }

    print(f"\n[5/5] Objective scores:")
    print(f"       V1  (toughness)    = {score_v1:.6f}")
    print(f"       V2R (target-dist)  = {score_v2:.6f}")

    _save_results(results, work)
    print(f"\n✓ Full pipeline test PASSED")
    print(f"  Results saved to: {work / 'results.json'}")
    return 0


def _save_results(results: dict, work: Path) -> None:
    out = work / "results.json"
    out.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
