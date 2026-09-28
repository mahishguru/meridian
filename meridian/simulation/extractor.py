"""Mechanical property extraction from a stress-strain CSV.

Logic mirrors `Extract_Results.py` but exposes a clean API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass
class MechanicalProperties:
    E: float                       # Young's modulus [MPa]
    sigma_y: float                 # Yield stress [MPa]
    epsilon_y: float               # Yield strain
    sigma_u: float                 # Ultimate tensile stress [MPa]
    epsilon_uniform: float         # Uniform strain at UTS
    K: float                       # Strength coefficient (Hollomon) [MPa]
    n: float                       # Hardening exponent (Hollomon)
    epsilon_fracture: float        # Fracture strain
    sigma_fracture: float          # Fracture stress [MPa]
    work_uniform: float = float("nan")          # ∫ sigma d epsilon up to UTS [MPa]
    work_fracture: float = float("nan")         # ∫ sigma d epsilon over full curve [MPa]
    plastic_work_uniform: float = float("nan")  # ∫ sigma d epsilon_p up to UTS [MPa]
    plastic_work_fracture: float = float("nan") # ∫ sigma d epsilon_p over full curve [MPa]
    n_grains: int = -1             # Number of grains in the RVE (-1 if unknown)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MechanicalProperties":
        """Build from a JSON dict, tolerating properties written by older runs."""
        names = {f.name for f in fields(cls)}
        kwargs = {k: d[k] for k in names if k in d}
        return cls(
            E=float(kwargs["E"]),
            sigma_y=float(kwargs["sigma_y"]),
            epsilon_y=float(kwargs["epsilon_y"]),
            sigma_u=float(kwargs["sigma_u"]),
            epsilon_uniform=float(kwargs["epsilon_uniform"]),
            K=float(kwargs["K"]),
            n=float(kwargs["n"]),
            epsilon_fracture=float(kwargs["epsilon_fracture"]),
            sigma_fracture=float(kwargs["sigma_fracture"]),
            work_uniform=float(kwargs.get("work_uniform", float("nan"))),
            work_fracture=float(kwargs.get("work_fracture", float("nan"))),
            plastic_work_uniform=float(kwargs.get("plastic_work_uniform", float("nan"))),
            plastic_work_fracture=float(kwargs.get("plastic_work_fracture", float("nan"))),
            n_grains=int(kwargs.get("n_grains", -1)),
        )


class PropertyExtractor:
    def __init__(self, youngs_strain_max: float = 1.0e-3, yield_tolerance: float = 0.4) -> None:
        self.youngs_strain_max = youngs_strain_max
        self.yield_tolerance = yield_tolerance

    @staticmethod
    def _youngs_modulus(strain: np.ndarray, stress: np.ndarray, eps_max: float) -> float:
        mask = strain < eps_max
        if mask.sum() < 2:
            mask = strain < strain[min(2, len(strain) - 1)]
        slope, _, _, _, _ = linregress(strain[mask], stress[mask])
        return float(slope)

    def _yield(self, strain: np.ndarray, stress: np.ndarray, E: float) -> tuple[float, float]:
        for i in range(1, len(strain)):
            if strain[i] == 0:
                continue
            Ei = stress[i] / strain[i]
            deviation = abs(Ei - E) / E
            if deviation > self.yield_tolerance:
                return float(strain[i]), float(stress[i])
        return float("nan"), float("nan")

    @staticmethod
    def _K_n(strain: np.ndarray, stress: np.ndarray, sigma_y: float) -> tuple[float, float]:
        idx_y = np.where(stress >= sigma_y)[0]
        if len(idx_y) == 0:
            return float("nan"), float("nan")
        start = idx_y[0]
        end = int(np.argmax(stress))
        if end - start < 3:
            return float("nan"), float("nan")
        s = stress[start:end]
        e = strain[start:end]
        true_s = s * (1.0 + e)
        true_e = np.log(1.0 + e)
        valid = (true_e > 0) & (true_s > 0)
        if valid.sum() < 3:
            return float("nan"), float("nan")
        slope, intercept, _, _, _ = linregress(np.log(true_e[valid]), np.log(true_s[valid]))
        return float(np.exp(intercept)), float(slope)

    @staticmethod
    def _fracture(strain: np.ndarray, stress: np.ndarray, E: float) -> tuple[float, float]:
        idx_u = int(np.argmax(stress))
        for i in range(idx_u + 1, len(stress) - 1):
            if stress[i] - stress[i + 1] > 1.0:
                sigma_f = float(stress[i])
                eps_f = float(strain[i] - sigma_f / E)
                return eps_f, sigma_f
        sigma_f = float(stress[-1])
        eps_f = float(strain[-1] - sigma_f / E)
        return eps_f, sigma_f

    @staticmethod
    def _work_terms(strain: np.ndarray, stress: np.ndarray, E: float, ult_idx: int) -> tuple[float, float, float, float]:
        """Engineering/plastic work densities from the stress-strain curve.

        Units are MPa because strain is dimensionless. These are objective
        features only; no constitutive calibration is changed.
        """
        if len(strain) < 2 or E <= 0 or not np.isfinite(E):
            nan = float("nan")
            return nan, nan, nan, nan
        ult_idx = max(1, min(int(ult_idx), len(strain) - 1))
        stress_pos = np.maximum(stress, 0.0)
        work_uniform = float(np.trapezoid(stress_pos[: ult_idx + 1], strain[: ult_idx + 1]))
        work_fracture = float(np.trapezoid(stress_pos, strain))

        plastic_strain = np.maximum(strain - stress_pos / E, 0.0)
        # Numerical noise can make the elastic-corrected strain locally non-
        # monotone, which would give negative trapezoids. Enforce monotonicity
        # for a robust scalar feature.
        plastic_strain = np.maximum.accumulate(plastic_strain)
        plastic_work_uniform = float(np.trapezoid(stress_pos[: ult_idx + 1], plastic_strain[: ult_idx + 1]))
        plastic_work_fracture = float(np.trapezoid(stress_pos, plastic_strain))
        return work_uniform, work_fracture, plastic_work_uniform, plastic_work_fracture

    def extract(self, csv_path: str | Path) -> Optional[MechanicalProperties]:
        df = pd.read_csv(csv_path)
        # Accept either old (no header) or new (with header) layout.
        if "strain_eng" in df.columns:
            strain = df["strain_eng"].to_numpy(dtype=float)
            stress = df["stress_eng"].to_numpy(dtype=float)
        else:
            arr = df.to_numpy(dtype=float)
            strain = arr[:, 1]
            stress = arr[:, 2]

        # Defensive: yield/K-n/fracture iterate positionally and assume
        # monotonically increasing strain. Older CSVs (and any future producer
        # that forgets numeric sort) may store rows in lex-of-increment order.
        order = np.argsort(strain, kind="stable")
        strain = strain[order]
        stress = stress[order]

        if len(strain) < 5 or np.max(stress) <= 0:
            return None

        ult_idx = int(np.argmax(stress))
        sigma_u = float(stress[ult_idx])
        eps_at_uts = float(strain[ult_idx])

        try:
            E = self._youngs_modulus(strain, stress, self.youngs_strain_max)
            if E <= 0 or not np.isfinite(E):
                return None
            eps_y, sigma_y = self._yield(strain, stress, E)
            if not np.isfinite(sigma_y):
                return None
            K, n = self._K_n(strain, stress, sigma_y)
            eps_uniform = eps_at_uts - sigma_u / E
            eps_f, sigma_f = self._fracture(strain, stress, E)
            work_u, work_f, pwork_u, pwork_f = self._work_terms(strain, stress, E, ult_idx)
        except Exception:
            return None

        return MechanicalProperties(
            E=E,
            sigma_y=sigma_y,
            epsilon_y=eps_y,
            sigma_u=sigma_u,
            epsilon_uniform=eps_uniform,
            K=K,
            n=n,
            epsilon_fracture=eps_f,
            sigma_fracture=sigma_f,
            work_uniform=work_u,
            work_fracture=work_f,
            plastic_work_uniform=pwork_u,
            plastic_work_fracture=pwork_f,
        )
