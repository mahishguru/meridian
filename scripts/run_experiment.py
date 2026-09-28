#!/usr/bin/env python
"""Convenience wrapper: `python scripts/run_experiment.py --config ...`."""
import sys

from meridian.cli import run_experiment

if __name__ == "__main__":
    sys.exit(run_experiment(sys.argv[1:]))
