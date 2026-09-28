"""Run a single DAMASK_grid simulation from a Dream3D file.

Distilled from the original `Run_batch.py` into a re-usable class.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DAMASKResult:
    sim_dir: Path
    hdf5_path: Path | None
    returncode: int
    log_path: Path


class DAMASKRunner:
    """Convert a .dream3d file to DAMASK inputs and run DAMASK_grid."""

    def __init__(
        self,
        load_yaml: str | Path,
        phase_yaml: str | Path,
        target_phase: str = "AZ31",
        target_homog: str = "SX",
        damask_binary: str = "DAMASK_grid",
        n_threads: int = 8,
        timeout_seconds: int = 7200,
        bind_cpus: bool = True,
    ) -> None:
        self.load_yaml = Path(load_yaml).resolve()
        self.phase_yaml = Path(phase_yaml).resolve()
        self.target_phase = target_phase
        self.target_homog = target_homog
        self.damask_binary = damask_binary
        self.n_threads = n_threads
        self.timeout_seconds = timeout_seconds
        self.bind_cpus = bind_cpus

        if not self.load_yaml.is_file():
            raise FileNotFoundError(self.load_yaml)
        if not self.phase_yaml.is_file():
            raise FileNotFoundError(self.phase_yaml)

    def _convert_dream3d(self, dream3d: Path, out_dir: Path) -> tuple[Path, Path]:
        import damask  # noqa: F401  -- imported lazily; heavy

        sim_name = dream3d.stem
        material_out = out_dir / f"{sim_name}_material.yaml"
        geom_out = out_dir / f"{sim_name}.vti"

        m = damask.ConfigMaterial().load_DREAM3D(
            fname=str(dream3d),
            grain_data="Grain Data",
            cell_data="CellData",
            cell_ensemble_data="CellEnsembleData",
            phases="Phases",
            Euler_angles="EulerAngles",
            phase_names="PhaseName",
            base_group="DataContainers/SyntheticVolumeDataContainer",
        )
        material_points = m["material"]
        phase_data = damask.ConfigMaterial.load(str(self.phase_yaml))
        m.clear()
        m["homogenization"] = {
            self.target_homog: {"N_constituents": 1, "mechanical": {"type": "pass"}}
        }
        m["phase"] = phase_data["phase"]
        m["material"] = material_points
        for mat in m["material"]:
            for c in mat["constituents"]:
                c["phase"] = self.target_phase
            mat["homogenization"] = self.target_homog
        m.save(fname=str(material_out))

        t = damask.GeomGrid.load_DREAM3D(
            fname=str(dream3d),
            feature_IDs="FeatureIds",
            cell_data="CellData",
            phases="Phases",
            Euler_angles="EulerAngles",
            base_group="DataContainers/SyntheticVolumeDataContainer",
        )
        t.save(fname=str(geom_out), compress=True)
        return material_out, geom_out

    def run(self, dream3d: str | Path, sim_dir: str | Path) -> DAMASKResult:
        dream3d = Path(dream3d).resolve()
        sim_dir = Path(sim_dir).resolve()
        sim_dir.mkdir(parents=True, exist_ok=True)

        material_file, geom_file = self._convert_dream3d(dream3d, sim_dir)
        shutil.copy(self.load_yaml, sim_dir / "load.yaml")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.n_threads)
        if self.bind_cpus:
            env["OMP_PROC_BIND"] = "TRUE"
            env["OMP_PLACES"] = "cores"
        else:
            # When several DAMASK subprocesses run concurrently, pinning would
            # cause them all to fight for the same physical cores. Let the OS
            # scheduler load-balance instead.
            env["OMP_PROC_BIND"] = "FALSE"
            env.pop("OMP_PLACES", None)
        env.setdefault("UCX_TLS", "shm,self")

        log_path = sim_dir / "damask.log"
        with log_path.open("w") as log_file:
            try:
                result = subprocess.run(
                    [
                        self.damask_binary,
                        "--geom", geom_file.name,
                        "--load", "load.yaml",
                        "--material", material_file.name,
                    ],
                    cwd=sim_dir,
                    stdout=log_file,
                    stderr=log_file,
                    env=env,
                    timeout=self.timeout_seconds,
                    check=False,
                    # Detach from the launcher's controlling TTY so a SIGHUP
                    # to the parent shell (ssh blip, terminal close, logind
                    # session bounce) does not propagate to DAMASK/PETSc.
                    # Without this, every concurrent simulation across every
                    # optimizer run dies simultaneously with rc=15.
                    start_new_session=True,
                )
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = -1

        hdf5 = next(sim_dir.glob("*.hdf5"), None)
        return DAMASKResult(sim_dir=sim_dir, hdf5_path=hdf5, returncode=returncode, log_path=log_path)

    @staticmethod
    def cleanup(sim_dir: str | Path) -> None:
        for f in Path(sim_dir).glob("*.hdf5"):
            try:
                f.unlink()
            except OSError:
                pass
