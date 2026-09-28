#!/usr/bin/env python
"""Build v3 seed caches from the existing v2 ones.

A seed cache stores ``Z`` (latents), ``Y`` (objective values) and ``success``.
``Y`` is *precomputed*, so handing a v2 cache to a v3 run would warm-start the
optimizer with 100 v2 scores and silently make the campaign incomparable --
exactly the class of bug the v2 audit turned up. The latents and the underlying
DAMASK simulations are unchanged, so only ``Y`` has to be recomputed.

The 100 seed rows are recovered from any completed v2 cell's ``log.jsonl``
(``iter == 0``, ordered by ``batch_idx``). That ordering is verified against the
cached ``Y`` before anything is written: the v2 objective evaluated on the
logged properties must reproduce the cached values bit-for-bit.

Usage:
    python scripts/make_v3_seed_caches.py            # write
    python scripts/make_v3_seed_caches.py --check    # verify only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meridian.config import apply_overrides, load_config  # noqa: E402
from meridian.objectives import get_objective  # noqa: E402

DECODERS = ["fmdit", "fmdit_768", "fmdit_1024", "fmdit_1280"]
CONFIG = {d: ROOT / "config" / f"mg5gd_{d}_meridian_v2.yaml" for d in DECODERS}
CACHE_DIR = ROOT / "results" / "seed_caches"
# Any completed v2 cell works; the seed rows are identical across cells.
REF_CELL = "{dec}__turbo__v2__r0"


class _Props:
    """Minimal stand-in for MechanicalProperties (needs .to_dict/.n_grains)."""

    def __init__(self, d: dict) -> None:
        self._d = d
        self.n_grains = int(d.get("n_grains", 0) or 0)

    def to_dict(self) -> dict:
        return self._d


def seed_props(dec: str) -> list[dict]:
    log = ROOT / f"results/Mg5Gd_v2_{dec}" / REF_CELL.format(dec=dec) / "log.jsonl"
    rows = [json.loads(line) for line in log.open()]
    seed = sorted((r for r in rows if r["iter"] == 0), key=lambda r: r["batch_idx"])
    return [r["props"] for r in seed]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Verify alignment and report, but write nothing.")
    args = ap.parse_args()

    for dec in DECODERS:
        src = CACHE_DIR / f"seed_cache_Mg5Gd_v2_{dec}.npz"
        dst = CACHE_DIR / f"seed_cache_Mg5Gd_v3_{dec}.npz"
        data = np.load(src, allow_pickle=True)
        Z, Y2, ok = data["Z"], data["Y"], data["success"]

        props = [_Props(p) for p in seed_props(dec)]
        if len(props) != len(Y2):
            print(f"ERROR: {dec}: {len(props)} logged seeds vs {len(Y2)} cached",
                  file=sys.stderr)
            return 1

        cfg = load_config(CONFIG[dec])
        obj_v2 = get_objective(cfg)
        obj_v3 = get_objective(apply_overrides(load_config(CONFIG[dec]),
                                               ["objective.name=v3"]))

        # Guard: the logged properties must regenerate the cached v2 scores in
        # this exact order, otherwise Z and Y would be mismatched by row.
        check = np.array([obj_v2(p) for p in props], dtype=np.float32)
        err = float(np.abs(check - Y2).max())
        if err > 1e-6:
            print(f"ERROR: {dec}: seed order mismatch, max|dY|={err:.3e}",
                  file=sys.stderr)
            return 1

        Y3 = np.array([obj_v3(p) for p in props], dtype=np.float32)
        print(f"{dec:<12} n={len(Y3):3d} align={err:.1e}  "
              f"v2 best={Y2.max():+.4f} mean={Y2.mean():+.4f}  ->  "
              f"v3 best={Y3.max():+.4f} mean={Y3.mean():+.4f}")

        if not args.check:
            np.savez(dst, Z=Z, Y=Y3, success=ok)
            print(f"{'':<12} wrote {dst.relative_to(ROOT)}")

    print("\ncheck only, nothing written" if args.check else "\nv3 seed caches ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
