"""Stress-strain extraction from DAMASK HDF5.

Lean re-implementation of `DAMASK_Post_Auto.py`. Returns a DataFrame with
columns matching the legacy CSV layout: `[step, strain_eng, stress_eng]`,
so downstream property extraction code stays unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _collect(node, key):
    """Descend nested damask.Result.get() dicts and collect arrays under key."""
    out = []
    if isinstance(node, dict):
        val = node.get(key)
        if isinstance(val, np.ndarray):
            out.append(val)
        else:
            for v in node.values():
                out.extend(_collect(v, key))
    return out


class StressStrainProcessor:
    """Compute engineering stress and strain from a DAMASK HDF5 result file."""

    def __init__(self, loading_direction: str = "x") -> None:
        self.axis = {"x": 0, "y": 1, "z": 2}[loading_direction.lower()]

    def process(self, hdf5_file: str | Path) -> pd.DataFrame:
        import damask

        result = damask.Result(str(hdf5_file))
        a = self.axis

        # IMPORTANT: increments are keyed "increment_0", "increment_1",
        # "increment_10", ... so alphabetical order is lexicographic, not
        # numeric. Downstream property extraction iterates positionally and
        # assumes monotone strain, so sort numerically.
        def _step_idx(name: str) -> int:
            tail = name.rsplit("_", 1)[-1]
            try:
                return int(tail)
            except ValueError:
                return -1

        steps = sorted(result.increments, key=_step_idx)

        # MEMORY: never call result.get("F") on the full result -- that
        # materialises every increment at once (~16 GB for 300x300x1250) and
        # the repeated multi-GB transients fragment the heap so RSS never
        # returns to the OS (observed ~200 GB/cell after ~35 iterations,
        # OOM-killed cells). Stream one increment at a time (~13 MB peak).
        records = []
        for i, step in enumerate(steps):
            data = result.view(increments=step).get(["F", "P"])
            F_parts = _collect(data, "F") if isinstance(data, dict) else []
            P_parts = _collect(data, "P") if isinstance(data, dict) else []
            if not F_parts or not P_parts:
                raise RuntimeError(
                    f"DAMASK HDF5 missing required fields F/P at {step}: "
                    f"{hdf5_file}")
            Fi = F_parts[0] if len(F_parts) == 1 else np.concatenate(F_parts)
            Pi = P_parts[0] if len(P_parts) == 1 else np.concatenate(P_parts)
            # sigma_Cauchy = (1/J) * P * F^T   (per voxel)
            # We only need the (a,a) component:
            #   sigma[a,a] = (sum_k P[a,k] * F[a,k]) / det(F)
            J = np.linalg.det(Fi)
            sigma_aa = np.einsum("nk,nk->n", Pi[:, a, :], Fi[:, a, :]) / J
            stress_eng = float(np.mean(sigma_aa)) / 1.0e6   # Pa -> MPa
            strain_eng = float(np.mean(Fi[:, a, a])) - 1.0
            records.append((i, strain_eng, stress_eng))

        df = pd.DataFrame(records, columns=["step", "strain_eng", "stress_eng"])
        return df

    def process_to_csv(self, hdf5_file: str | Path, csv_path: str | Path) -> Path:
        df = self.process(hdf5_file)
        csv_path = Path(csv_path)
        df.to_csv(csv_path, index=False)
        return csv_path
