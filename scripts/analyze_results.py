#!/usr/bin/env python
"""Build ablation tables and convergence plots from a `results/ablation/` tree.

Outputs:
  - results/ablation/table_v1.csv, table_v1.tex
  - results/ablation/table_v2.csv, table_v2.tex
  - results/ablation/convergence.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_log(jsonl: Path) -> pd.DataFrame:
    rows = []
    for line in jsonl.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def best_so_far(y: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    cur = -np.inf
    for i, v in enumerate(y):
        cur = max(cur, v)
        out[i] = cur
    return out


def collect(root: Path) -> pd.DataFrame:
    records = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        log = run_dir / "log.jsonl"
        if not log.exists():
            continue
        parts = run_dir.name.split("__")
        if len(parts) < 3:
            continue
        decoder, optimizer, objective = parts[0], parts[1], parts[2]
        df = _read_log(log)
        df["decoder"] = decoder
        df["optimizer"] = optimizer
        df["objective"] = objective
        df["run"] = run_dir.name
        df["best_so_far"] = best_so_far(df["objective"].to_numpy(dtype=float))
        records.append(df)
    if not records:
        raise FileNotFoundError(f"No runs with log.jsonl found under {root}")
    return pd.concat(records, ignore_index=True)


def make_tables(df: pd.DataFrame, out_dir: Path) -> None:
    for obj in df["objective"].unique():
        sub = df[df["objective"] == obj]
        # Best objective per (decoder, optimizer), averaged across repeats
        agg = (
            sub.groupby(["decoder", "optimizer", "run"])["best_so_far"].max()
            .reset_index()
            .groupby(["decoder", "optimizer"])["best_so_far"]
            .agg(["mean", "std"])
            .reset_index()
        )
        pivot_mean = agg.pivot(index="decoder", columns="optimizer", values="mean")
        pivot_std = agg.pivot(index="decoder", columns="optimizer", values="std")
        out_csv = out_dir / f"table_{obj}.csv"
        pivot_mean.to_csv(out_csv)
        # LaTeX with mean ± std
        cells = pivot_mean.copy().astype(object)
        for r in pivot_mean.index:
            for c in pivot_mean.columns:
                m = pivot_mean.loc[r, c]
                s = pivot_std.loc[r, c]
                cells.loc[r, c] = f"{m:.3f} \\pm {0.0 if pd.isna(s) else s:.3f}"
        tex = cells.to_latex(escape=False, na_rep="--", caption=f"Ablation (objective {obj})", label=f"tab:ablation_{obj}")
        (out_dir / f"table_{obj}.tex").write_text(tex)


def make_convergence(df: pd.DataFrame, out_path: Path) -> None:
    decoders = sorted(df["decoder"].unique())
    objectives = sorted(df["objective"].unique())
    fig, axes = plt.subplots(len(objectives), len(decoders), figsize=(4 * len(decoders), 3 * len(objectives)),
                              sharex=True, squeeze=False)
    for i, obj in enumerate(objectives):
        for j, dec in enumerate(decoders):
            ax = axes[i][j]
            sub = df[(df["objective"] == obj) & (df["decoder"] == dec)]
            for opt, g in sub.groupby("optimizer"):
                # Average best_so_far across repeats, indexed by evaluation count
                arrs = []
                for run, gg in g.groupby("run"):
                    arrs.append(gg["best_so_far"].to_numpy())
                if not arrs:
                    continue
                L = min(len(a) for a in arrs)
                A = np.stack([a[:L] for a in arrs])
                ax.plot(np.arange(L), A.mean(0), label=opt)
            ax.set_title(f"{dec} | {obj}")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel("best so far")
            if i == len(objectives) - 1:
                ax.set_xlabel("evaluations")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str, default="results/ablation",
                   help="Directory containing per-run subfolders.")
    args = p.parse_args()
    root = Path(args.results_root)
    df = collect(root)
    make_tables(df, root)
    make_convergence(df, root / "convergence.png")
    print(f"Wrote tables and convergence.png under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
