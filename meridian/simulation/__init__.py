"""End-to-end simulation pipeline facade.

Composes: latent z (already decoded by caller) -> PNG -> Dream3D
        -> DAMASK -> stress/strain CSV -> mechanical properties.

The decoder step is intentionally NOT here: the optimization loop owns
the decoder so it can re-use a single loaded model across many calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

from PIL import Image

from meridian.simulation.codec import png_to_dream3d, recon_to_dream3d
from meridian.simulation.damask import DAMASKRunner
from meridian.simulation.extractor import MechanicalProperties, PropertyExtractor
from meridian.simulation.postproc import StressStrainProcessor


@dataclass
class SimulationOutcome:
    success: bool
    properties: Optional[MechanicalProperties]
    sim_dir: Optional[Path]
    error: Optional[str] = None


class SimulationPipeline:
    """PNG -> Dream3D -> DAMASK -> properties."""

    def __init__(
        self,
        damask: DAMASKRunner,
        postproc: StressStrainProcessor,
        extractor: PropertyExtractor,
        class_means_json: str | Path,
        class_key: str,
        template_dream3d: str | Path | None = None,
        colour_tol: "int | str | None" = None,
        spacing: "np.ndarray | None" = None,
        cleanup_hdf5: bool = True,
        min_grains: int | None = None,
        target_size: int | None = None,
        min_pixel_std: float | None = None,
        calibrate: bool = False,
        texture_prior_dir: "str | Path | None" = None,
    ) -> None:
        self.damask = damask
        self.postproc = postproc
        self.extractor = extractor
        self.class_means_json = Path(class_means_json)
        self.class_key = class_key
        self.template_dream3d = Path(template_dream3d) if template_dream3d else None
        self.colour_tol = colour_tol
        self.spacing = spacing
        self.cleanup_hdf5 = cleanup_hdf5
        self.min_grains = int(min_grains) if min_grains is not None else None
        self.target_size = int(target_size) if target_size is not None else None
        self.min_pixel_std = float(min_pixel_std) if min_pixel_std is not None else None
        # Mandatory z-only ODF calibration (calibPW) for diffusion-decoder
        # outputs: SAM grains + whole-grain Sinkhorn repaint onto the
        # texture-head-predicted histogram, then png_to_dream3d(colour_tol=2).
        # Raw recon PNGs have a class-mean-collapsed ODF and must NEVER be
        # fed to png_to_dream3d directly when calibrate=True is configured.
        self.calibrate = bool(calibrate)
        self.texture_prior = None
        if self.calibrate:
            if texture_prior_dir is None:
                raise ValueError("codec.calibrate=true requires codec.texture_prior_dir")
            from meridian.simulation.texture import TexturePrior
            self.texture_prior = TexturePrior(texture_prior_dir)

    def _prepare_calibrated_png(self, png_path: "Path", work_dir: "Path",
                                z: "np.ndarray") -> "Path":
        """Raw decoder PNG -> NEAREST 300^2 -> blank check -> calibPW PNG."""
        import numpy as np
        from PIL import Image as _PILImage

        img = _PILImage.open(png_path).convert("RGB")
        if self.target_size is not None and img.size != (self.target_size,) * 2:
            t = int(self.target_size)
            resized = work_dir / f"{png_path.stem}__r{t}.png"
            img = img.resize((t, t), _PILImage.NEAREST)
            img.save(resized)
            png_path = resized
        if self.min_pixel_std is not None and float(self.min_pixel_std) > 0.0:
            arr = np.asarray(img, dtype=np.float32)
            std = float(arr.reshape(-1, 3).std(axis=0).mean())
            if std < float(self.min_pixel_std):
                raise ValueError(
                    f"blank_decode: pixel_std={std:.2f} < "
                    f"min_pixel_std={float(self.min_pixel_std):.2f}")
        hist = self.texture_prior.predict_hist(z)
        hist_npz = work_dir / "pred_hist.npz"
        np.savez(hist_npz, **{png_path.stem: hist})
        return png_path, hist_npz

    def evaluate_png(self, png_path: str | Path, work_dir: str | Path,
                     z: "np.ndarray | None" = None) -> SimulationOutcome:
        png_path = Path(png_path)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        n_grains = -1
        try:
            dream3d_out = work_dir / f"{png_path.stem}.dream3d"
            if self.calibrate:
                if z is None:
                    raise ValueError("calibrated pipeline needs the latent z")
                png300, hist_npz = self._prepare_calibrated_png(
                    png_path, work_dir, z)
                dream3d_out = work_dir / f"{png300.stem}.dream3d"
                _, n_grains = recon_to_dream3d(
                    recon_png=png300,
                    class_means_json=self.class_means_json,
                    class_key=self.class_key,
                    output_dream3d=dream3d_out,
                    pred_hist_npz=hist_npz,
                    sample_id=png300.stem,
                    spacing=self.spacing,
                    min_grains=self.min_grains,
                )
                calibrated_png = png300.with_name(
                    f"{png300.stem}_calibPW_sam.png")
                publish_tmp = png_path.with_name(
                    f".{png_path.name}.calibrated.tmp")
                shutil.copyfile(calibrated_png, publish_tmp)
                publish_tmp.replace(png_path)
            else:
                _, n_grains = png_to_dream3d(
                    png_path=png_path,
                    class_means_json=self.class_means_json,
                    class_key=self.class_key,
                    output_dream3d=dream3d_out,
                    template_dream3d=self.template_dream3d,
                    colour_tol=self.colour_tol,
                    spacing=self.spacing,
                    min_grains=self.min_grains,
                    target_size=self.target_size,
                    min_pixel_std=self.min_pixel_std,
                )
        except Exception as exc:
            return SimulationOutcome(False, None, work_dir, f"codec_failed: {exc!r}")

        # HDF5s are ~19 GB each; leaking them on failure paths fills the shared
        # filesystem and cascades into further failures.
        try:
            try:
                res = self.damask.run(dream3d_out, sim_dir=work_dir)
            except Exception as exc:
                return SimulationOutcome(False, None, work_dir, f"damask_launch_failed: {exc!r}")
            if res.returncode != 0 or res.hdf5_path is None:
                return SimulationOutcome(False, None, work_dir, f"damask_failed: rc={res.returncode}")

            try:
                csv_path = work_dir / "damask_stress_strain.csv"
                self.postproc.process_to_csv(res.hdf5_path, csv_path)
                props = self.extractor.extract(csv_path)
            except Exception as exc:
                return SimulationOutcome(False, None, work_dir, f"postproc_failed: {exc!r}")
        finally:
            if self.cleanup_hdf5:
                DAMASKRunner.cleanup(work_dir)

        if props is None:
            return SimulationOutcome(False, None, work_dir, "extraction_returned_none")

        # Stamp the grain count onto the extracted properties so downstream
        # objectives can apply representativeness penalties.
        try:
            props.n_grains = int(n_grains)
        except Exception:
            pass
        return SimulationOutcome(True, props, work_dir)

    def save_image(self, image: Image.Image, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path
