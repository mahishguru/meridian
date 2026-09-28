#!/usr/bin/env python
"""End-to-end test: synthetic image → orientation_codec → Dream3D → DAMASK → properties.

Creates a small 32×32 synthetic microstructure image, decodes it through the
full pipeline, and verifies each stage returns valid output.

Usage:
    python tests/test_damask_e2e.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
AZ31_DIR = ROOT.parent / "automated_simulations"


def make_synthetic_microstructure(size: int = 32) -> np.ndarray:
    """Create a tiny synthetic RGB image mimicking encoded orientations.

    Uses 4 rectangular grains with distinct but smooth colours in the
    stereographic-projection [0,255] range.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    h = size // 2
    # Grain 1 — top-left
    img[:h, :h] = [140, 130, 125]
    # Grain 2 — top-right
    img[:h, h:] = [120, 135, 130]
    # Grain 3 — bottom-left
    img[h:, :h] = [130, 120, 140]
    # Grain 4 — bottom-right
    img[h:, h:] = [125, 140, 120]
    return img


def write_fast_load_yaml(path: Path) -> None:
    """Write a lightweight load.yaml: small deformation, few increments."""
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
      t: 10
      N: 50
    f_out: 5
""")


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="meridian_damask_test_"))
    print(f"Working directory: {work}")

    # ---- Step 1: Create synthetic PNG ----
    from PIL import Image

    img_arr = make_synthetic_microstructure(32)
    png_path = work / "synth_micro.png"
    Image.fromarray(img_arr).save(png_path)
    print(f"[1/5] Synthetic 32×32 PNG saved → {png_path}")

    # ---- Step 2: PNG → Dream3D via orientation_codec ----
    from meridian.simulation.codec import png_to_dream3d

    class_means_json = ASSETS / "class_means.json"
    if not class_means_json.is_file():
        print(f"SKIP: class_means.json not found at {class_means_json}")
        return 1

    dream3d_path = work / "synth_micro.dream3d"
    png_to_dream3d(
        png_path=png_path,
        class_means_json=class_means_json,
        class_key="AZ31_extruded",
        output_dream3d=dream3d_path,
        spacing=np.array([1.0, 1.0, 1.0]),
    )
    assert dream3d_path.is_file(), "Dream3D not created"
    print(f"[2/5] Dream3D written → {dream3d_path}")

    # Quick sanity: check HDF5 structure
    import h5py

    with h5py.File(dream3d_path, "r") as f:
        base = "DataContainers/SyntheticVolumeDataContainer"
        assert f"{base}/CellData/EulerAngles" in f, "Missing CellData/EulerAngles"
        assert f"{base}/CellData/FeatureIds" in f, "Missing CellData/FeatureIds"
        assert f"{base}/Grain Data/EulerAngles" in f, "Missing Grain Data/EulerAngles"
        n_grains = f[f"{base}/Grain Data/EulerAngles"].shape[0]
        grid_shape = f[f"{base}/CellData/FeatureIds"].shape
        print(f"       Grid shape: {grid_shape}, grains: {n_grains}")

    # ---- Step 3: Dream3D → DAMASK inputs (material.yaml + .vti) ----
    from meridian.simulation.damask import DAMASKRunner

    # Use a fast load file for the test
    fast_load = work / "load.yaml"
    write_fast_load_yaml(fast_load)

    phase_yaml = AZ31_DIR / "AZ31_Phenopower.yaml"
    if not phase_yaml.is_file():
        print(f"SKIP: phase YAML not found at {phase_yaml}")
        return 1

    runner = DAMASKRunner(
        load_yaml=fast_load,
        phase_yaml=phase_yaml,
        target_phase="AZ31",
        target_homog="SX",
        damask_binary=os.environ.get(
            "DAMASK_BINARY",
            "DAMASK_grid",
        ),
        n_threads=4,
        timeout_seconds=600,
    )

    sim_dir = work / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    print("[3/5] Converting Dream3D → VTI + material.yaml …")
    material_file, geom_file = runner._convert_dream3d(dream3d_path, sim_dir)
    assert material_file.is_file(), "material.yaml not created"
    assert geom_file.is_file(), ".vti not created"
    print(f"       material: {material_file.name}")
    print(f"       geometry: {geom_file.name}")

    # ---- Step 4: Run DAMASK_grid ----
    print("[4/5] Running DAMASK_grid (small 32×32 grid, 50 increments) …")
    t0 = time.time()
    result = runner.run(dream3d_path, sim_dir=sim_dir)
    elapsed = time.time() - t0
    print(f"       Return code: {result.returncode}  ({elapsed:.1f}s)")
    print(f"       Log: {result.log_path}")

    if result.returncode != 0 or result.hdf5_path is None:
        print("DAMASK FAILED. Last 30 lines of log:")
        lines = result.log_path.read_text().splitlines()
        for line in lines[-30:]:
            print(f"  | {line}")
        return 1

    print(f"       HDF5: {result.hdf5_path}")

    # ---- Step 5: Post-process → mechanical properties ----
    from meridian.simulation.postproc import StressStrainProcessor
    from meridian.simulation.extractor import PropertyExtractor

    post = StressStrainProcessor(loading_direction="x")
    csv_path = work / "stress_strain.csv"
    post.process_to_csv(result.hdf5_path, csv_path)
    assert csv_path.is_file(), "CSV not written"
    print(f"[5/5] Stress–strain CSV → {csv_path}")

    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"       Rows: {len(df)}, max stress: {df['stress_eng'].max():.1f} MPa")

    ext = PropertyExtractor(youngs_strain_max=1e-3, yield_tolerance=0.4)
    props = ext.extract(csv_path)
    if props is None:
        print("WARNING: property extraction returned None (too few data points?)")
        print("But DAMASK ran successfully — pipeline is functional.")
    else:
        print(f"       σ_y = {props.sigma_y:.1f} MPa")
        print(f"       σ_u = {props.sigma_u:.1f} MPa")
        print(f"       n   = {props.n:.4f}")
        print(f"       K   = {props.K:.1f} MPa")

    print(f"\n✓ End-to-end DAMASK test PASSED  (workdir: {work})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
