#!/usr/bin/env python
"""Render grain-structure PNGs from .vti or .dream3d files.

Usage:
    render_grain_png.py FILE [FILE ...] [--out DIR] [--slice mid|N] [--axis 0|1|2]
                                        [--source SOURCE_PNG] [--field FeatureIds]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image


def _read_vti(path: Path, field: str) -> np.ndarray:
    import vtk
    from vtk.util import numpy_support as ns
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path)); r.Update()
    img = r.GetOutput()
    dims = img.GetDimensions()  # (nx,ny,nz) of points; cells = dim-1
    cd = img.GetCellData()
    arr = cd.GetArray(field)
    if arr is None:
        names = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
        raise KeyError(f"Field '{field}' not in {path.name}. Available: {names}")
    a = ns.vtk_to_numpy(arr)
    cell_dims = (max(1, dims[0]-1), max(1, dims[1]-1), max(1, dims[2]-1))
    # VTK is x-fastest; reshape as (nz,ny,nx)
    return a.reshape(cell_dims[2], cell_dims[1], cell_dims[0])


def _read_dream3d(path: Path, field: str) -> np.ndarray:
    import h5py
    dc = "DataContainers/SyntheticVolumeDataContainer/CellData"
    with h5py.File(path, "r") as f:
        a = f[f"{dc}/{field}"][...]
    return np.asarray(a).squeeze()


def _slice(vol: np.ndarray, axis: int, idx) -> np.ndarray:
    if vol.ndim == 2:
        return vol
    if vol.ndim == 4:  # last dim is comp
        vol = vol[..., 0]
    n = vol.shape[axis]
    if idx == "mid":
        idx = n // 2
    else:
        idx = int(idx)
    return np.take(vol, idx, axis=axis)


def _colorise(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(labels.max()) + 1
    cmap = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    cmap[0] = 0  # background
    return cmap[labels.astype(np.int64)]


def render(path: Path, out_dir: Path, slc, axis: int, field: str,
           source_png: Path | None = None) -> Path:
    if path.suffix == ".vti":
        vol = _read_vti(path, field)
    elif path.suffix == ".dream3d":
        vol = _read_dream3d(path, field)
    else:
        raise ValueError(f"Unsupported file: {path}")

    s2d = _slice(vol, axis, slc)
    n_grains = len(np.unique(s2d)) - (1 if 0 in s2d else 0)
    img = Image.fromarray(_colorise(s2d.astype(np.int64)))
    out = out_dir / f"{path.stem}_{field}.png"
    img.save(out)
    print(f"{path.name}: shape={s2d.shape}  n_grains={n_grains}  -> {out}")

    if source_png is not None:
        src = Image.open(source_png).convert("RGB").resize(s2d.shape[::-1], Image.NEAREST)
        sbs = Image.new("RGB", (s2d.shape[1] * 2 + 10, s2d.shape[0]), (255, 255, 255))
        sbs.paste(src, (0, 0)); sbs.paste(img, (s2d.shape[1] + 10, 0))
        sbs_path = out_dir / f"{path.stem}_compare.png"
        sbs.save(sbs_path)
        print(f"  side-by-side: {sbs_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path.cwd())
    ap.add_argument("--slice", default="mid")
    ap.add_argument("--axis", type=int, default=0)
    ap.add_argument("--field", default="FeatureIds")
    ap.add_argument("--source", type=Path, default=None,
                    help="Optional source PNG for side-by-side compare")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        render(f, args.out, args.slice, args.axis, args.field, args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
