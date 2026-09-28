"""Latent -> PNG -> Dream3D bridge using ``orientation_codec``.

Two conversion strategies are supported (selected automatically):

1. **Segmentation-based** (default / ``template_dream3d=None``):
   Uses ``orientation_codec.decode_image_pixelwise`` which segments the
   decoded PNG, computes per-grain median orientations, and writes a full
   ``.dream3d`` with ``CellData``, ``Grain Data``, and ``CellEnsembleData``
   — directly loadable by ``damask.ConfigMaterial.load_DREAM3D``.
   This faithfully preserves the decoder's grain sizes and aspect ratios.

2. **Template-based** (legacy / ``template_dream3d`` provided):
   Copy an existing ``.dream3d`` and overwrite only the EulerAngles dataset.
   Grain morphology is **fixed** to the template's FeatureIds.
   Only appropriate when the decoder is guaranteed to produce the same
   grain structure as the template.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np


# SAM grain segmentation lives in microstructure_ed so the reconstruction
# metrics, the ODF calibration and this bridge share one implementation.
from microstructure_ed.segmentation import (  # noqa: E402,F401
    _compact_labels,
    _get_sam_amg,
    _sam_segment,
)


def _slic_segment(
    img: np.ndarray,
    n_segments: int = 180,
    compactness: float = 5.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """SLIC superpixel segmentation — preferred for FM-DiT / diffusion outputs.

    SLIC clusters pixels by colour + spatial proximity, producing roughly
    equally-sized superpixels whose count is set by ``n_segments``.
    Validated round-trip:
      - GT sim_95 (truth=139): n=137 (99% recovery, topology preserved)
      - FM-DiT 512x512 batch:  ~134 grains, boundaries follow source blobs
    """
    from skimage.segmentation import slic
    lbl = slic(
        img, n_segments=int(n_segments), compactness=float(compactness),
        sigma=float(sigma), channel_axis=-1, start_label=1,
    )
    return _compact_labels(lbl.astype(np.int32))


def _smart_segment(
    img: np.ndarray,
    n_colors: int = 32,
    median_size: int = 5,
    min_grain_size: int = 50,
) -> np.ndarray:
    """Robust segmentation for blurry/noisy generator outputs.

    Pipeline:
      1) PIL median filter (denoise speckle, preserve edges)
      2) PIL.quantize MEDIANCUT to ``n_colors`` (collapse smooth gradients
         into discrete colour bands)
      3) per-colour connected components
      4) drop components < ``min_grain_size`` pixels and Voronoi-fill from
         the surviving large grains so every voxel gets a label.

    Validated on AZ31 RVE round-trip (sim_95: 139 GT -> 111 recovered) and
    on FM-DiT 512x512 outputs (108-181 grains across a perturbed batch).
    """
    from PIL import Image, ImageFilter
    from scipy.ndimage import label as ndlabel, distance_transform_edt

    pil = Image.fromarray(img).filter(ImageFilter.MedianFilter(size=int(median_size)))
    q = np.array(pil.quantize(colors=int(n_colors), method=Image.Quantize.MEDIANCUT))

    label_map = np.zeros_like(q, dtype=np.int32)
    nid = 0
    for c in range(int(n_colors)):
        mask = (q == c)
        if not mask.any():
            continue
        cc, nc = ndlabel(mask)
        for i in range(1, nc + 1):
            nid += 1
            label_map[cc == i] = nid

    if int(min_grain_size) <= 1:
        return _compact_labels(label_map)

    sizes = np.bincount(label_map.ravel())
    new_lab = np.zeros_like(label_map)
    new_id = 0
    for lid in range(1, len(sizes)):
        if sizes[lid] >= int(min_grain_size):
            new_id += 1
            new_lab[label_map == lid] = new_id
    if new_lab.max() == 0:
        return _compact_labels(label_map)
    inv = (new_lab == 0)
    if inv.any():
        _, (yy, xx) = distance_transform_edt(inv, return_indices=True)
        new_lab = new_lab[yy, xx]
    return _compact_labels(new_lab)


def recon_to_dream3d(
    recon_png: "str | Path",
    class_means_json: "str | Path",
    class_key: str,
    output_dream3d: "str | Path",
    pred_hist_npz: "str | Path",
    sample_id: "str | None" = None,
    spacing=None,
    min_grains: "int | None" = 50,
) -> "tuple[Path, int]":
    """MANDATORY entry point for FM-DiT decoder outputs -> DAMASK.

    Raw decoder PNGs must NEVER be passed to png_to_dream3d directly: the
    ODE rollout collapses texture sharpness toward the class mean, so the
    per-image ODF (pole figures) is wrong until calibrated. This function
    always applies the full validated inference chain:

        z -> FM-DiT recon -> SAM grains -> Sinkhorn whole-grain repaint
          -> calibPW_sam.png -> png_to_dream3d -> DAMASK

    Steps (all z-only, no ground truth needed):
      1. SAM grain map of the recon (identical segmenter to png_to_dream3d,
         cached as <stem>_samlbl.npy).
      2. Whole-grain Sinkhorn repaint onto the predicted ODF (texture head
         histogram from pred_hist_npz) - preserves grain size/aspect ratio,
         fixes the ODF (EMD ~1-3 deg vs ~3-11 deg raw).
      3. png_to_dream3d on the flat-coloured calibrated PNG (colour_tol=2
         recovers exactly the painted SAM grains; SAM is not re-run).

    pred_hist_npz: npz of per-sample predicted histograms (texture_prior.py
    predict); sample_id selects the entry (optional if npz has exactly one).
    Returns (dream3d_path, n_grains) like png_to_dream3d.
    """
    from microstructure_ed.eval.odf_calibrate import calibrate_recon_png  # deferred (heavy import)
    calib_png = calibrate_recon_png(recon_png, pred_hist_npz, sample_id=sample_id)
    return png_to_dream3d(calib_png, class_means_json, class_key, output_dream3d,
                          colour_tol=2, spacing=spacing, min_grains=min_grains)


def png_to_dream3d(
    png_path: str | Path,
    class_means_json: str | Path,
    class_key: str,
    output_dream3d: str | Path,
    template_dream3d: str | Path | None = None,
    colour_tol: "int | str | None" = None,
    spacing: np.ndarray | None = None,
    min_grains: int | None = None,
    target_size: int | None = None,
    min_pixel_std: float | None = None,
) -> tuple[Path, int]:
    """Decode a generated PNG into a DAMASK-ready Dream3D file.\n\n    WARNING: for FM-DiT decoder outputs use recon_to_dream3d instead, which\n    first applies the mandatory z-only ODF calibration (SAM grains + Sinkhorn\n    whole-grain repaint). Raw recon PNGs have a class-mean-collapsed ODF.

    Parameters
    ----------
    png_path : path to 8-bit PNG (or 16-bit TIFF) from the decoder
    class_means_json : path to class_means.json
    class_key : alloy key inside class_means.json
    output_dream3d : output .dream3d path
    template_dream3d : (optional) if provided, uses legacy template-copy path
    colour_tol : segmentation colour tolerance (only used in segmentation mode)
    spacing : (3,) voxel spacing; default [1,1,1]
    min_grains : (optional) reject (raise ``ValueError``) when the segmenter
        produces fewer than this many grains. Acts as a representativeness
        floor — under-resolved RVEs (e.g. 8–10 grains) give physically
        meaningless / texture-fluke responses that the optimizer would
        otherwise exploit because the constitutive law is grain-size
        insensitive.
    target_size : (optional) if set, the input PNG is NEAREST-downsampled to
        ``(target_size, target_size)`` before segmentation/decoding. Use
        together with ``spacing`` to keep the physical domain constant
        (e.g. target_size=300 + spacing=[2,2,2] reproduces the 600x600 µm
        AZ31 seed-cache RVEs from FM-DiT 512x512 outputs).
    min_pixel_std : (optional) reject (raise ``ValueError("blank_decode: ...")``)
        when the (resized) input PNG's per-channel std-dev mean is below
        this threshold. Catches the "watercolor-blob" decoder regime
        (smooth low-saturation images with only a handful of visible
        regions) *before* segmentation and DAMASK, so the optimizer gets
        a separate failure label distinct from ``too_few_grains``.

    Returns
    -------
    (path, n_grains) : path to the saved Dream3D file and the number of grains
        in the RVE (counted from the label_map for segmentation mode, or from
        ``mean_q.shape[0]`` for template mode if not derivable).
    """
    from orientation_codec import load_class_means

    png_path = Path(png_path)
    output_dream3d = Path(output_dream3d)
    output_dream3d.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optional NEAREST downsample of the input PNG so the polycrystal
    # delivered to DAMASK matches the seed-cache pipeline (300x300 @
    # 2 um -> 600 x 600 um physical domain). FM-DiT decodes 512x512 from
    # 300x300 training PNGs, so this just reverses the model-side upsample.
    # NEAREST preserves grain-ID topology exactly (no color blending).
    # ------------------------------------------------------------------
    if target_size is not None and int(target_size) > 0:
        from PIL import Image as _PILImage
        with _PILImage.open(png_path) as _im:
            tw, th = int(target_size), int(target_size)
            if _im.size != (tw, th):
                resized_path = output_dream3d.parent / f"{png_path.stem}__r{tw}.png"
                _im.convert("RGB").resize((tw, th), _PILImage.NEAREST).save(resized_path)
                png_path = resized_path

    # ------------------------------------------------------------------
    # Pre-codec sanity gate: reject "blank decode" (watercolor-blob)
    # outputs before paying the segmentation + DAMASK cost. The decoder
    # has three regimes for AZ31: blank (~5-25 fake grains, low pixel
    # std), AZ31-like (~100-300 grains, std ~ 35-40), and speckle
    # (~400+ spurious grains, std ~ 38-42). The blank regime evades
    # ``min_grains`` only when the segmenter happens to find boundaries,
    # but is far cheaper to reject upstream by image statistics. The
    # speckle regime is handled by the upper-band penalty in the
    # objectives, not here.
    # ------------------------------------------------------------------
    if min_pixel_std is not None and float(min_pixel_std) > 0.0:
        from PIL import Image as _PILImage
        with _PILImage.open(png_path) as _im:
            arr = np.asarray(_im.convert("RGB"), dtype=np.float32)
        std = float(arr.std(axis=(0, 1)).mean())
        if std < float(min_pixel_std):
            raise ValueError(
                f"blank_decode: pixel_std={std:.2f} < min_pixel_std={float(min_pixel_std):.2f}"
            )

    means = load_class_means(str(class_means_json))
    if class_key not in means:
        raise KeyError(
            f"class_key={class_key!r} not in class_means.json "
            f"(available: {list(means.keys())})"
        )
    mean_q = np.asarray(means[class_key], dtype=np.float64)

    if template_dream3d is not None:
        # --- Legacy path: overwrite Euler angles in a template ----------
        from orientation_codec.dataset import decode_pixelwise, load_png, load_16bit_tiff
        from orientation_codec.io_utils import copy_dream3d_with_new_eulers

        if png_path.suffix.lower() == ".png":
            img = load_png(png_path)
        else:
            img = load_16bit_tiff(png_path)

        euler_field = decode_pixelwise(img, mean_q)
        copy_dream3d_with_new_eulers(
            source_path=str(template_dream3d),
            output_path=str(output_dream3d),
            new_euler_angles=euler_field,
        )
        # Template mode: grain count is fixed by the template; read it back
        # from the file so callers still get a meaningful number.
        try:
            import h5py
            with h5py.File(output_dream3d, "r") as _f:
                n_grains_out = [None]
                def _cb(name, obj):
                    if name.endswith("Grain Data/EulerAngles") and n_grains_out[0] is None:
                        n_grains_out[0] = int(obj.shape[0])
                _f.visititems(_cb)
            n_grains = int(n_grains_out[0]) if n_grains_out[0] else -1
        except Exception:
            n_grains = -1
    else:
        # --- Segmentation path -----------------------------------------
        # Modes (selected by ``colour_tol``):
        #   - "sam" / "sam:PPS:IOU:MIN": Segment Anything Model (ViT-B)
        #     AutomaticMaskGenerator. Best topology fidelity for diffusion
        #     outputs; ~3 s/image on GPU. Defaults: pps=32, iou=0.86, min=200.
        #   - "slic" / "slic:NSEG:COMPACT": SLIC superpixels. Fast and
        #     robust; defaults: n=180, cp=5.
        #   - "smart" / "smart:NCOL:MED:MIN": median-filter + PIL.quantize
        #     + per-colour CC + tiny-grain Voronoi-fill. Legacy.
        #   - integer N: orientation_codec.segment_grains(colour_tol=N).
        #   - None: fall through to decode_image_pixelwise's default
        #     segment_grains (tol=1). Only valid for clean encoder PNGs.
        from orientation_codec.dataset import (
            decode_image_pixelwise,
            segment_grains,
            load_png,
            load_16bit_tiff,
        )

        img = load_png(png_path) if png_path.suffix.lower() == ".png" \
            else load_16bit_tiff(png_path)

        label_map = None
        if isinstance(colour_tol, str) and colour_tol.startswith("sam"):
            parts = colour_tol.split(":")
            pps = int(parts[1]) if len(parts) > 1 and parts[1] else 32
            iou = float(parts[2]) if len(parts) > 2 and parts[2] else 0.86
            mma = int(parts[3]) if len(parts) > 3 and parts[3] else 200
            label_map = _sam_segment(
                img, points_per_side=pps,
                pred_iou_thresh=iou, min_mask_region_area=mma,
            )
        elif isinstance(colour_tol, str) and colour_tol.startswith("slic"):
            parts = colour_tol.split(":")
            ns = int(parts[1]) if len(parts) > 1 and parts[1] else 180
            cp = float(parts[2]) if len(parts) > 2 and parts[2] else 5.0
            label_map = _slic_segment(img, n_segments=ns, compactness=cp)
        elif isinstance(colour_tol, str) and colour_tol.startswith("smart"):
            parts = colour_tol.split(":")
            n_colors = int(parts[1]) if len(parts) > 1 and parts[1] else 32
            med = int(parts[2]) if len(parts) > 2 and parts[2] else 5
            mins = int(parts[3]) if len(parts) > 3 and parts[3] else 50
            label_map = _smart_segment(
                img, n_colors=n_colors, median_size=med, min_grain_size=mins,
            )
        elif colour_tol is not None:
            label_map = segment_grains(img, colour_tol=int(colour_tol))

        # ---- Representativeness floor (raise BEFORE writing dream3d) ----
        # Only enforce when we actually have a label_map to count from;
        # the fallback (label_map is None) leaves enforcement to the
        # downstream call after decode_image_pixelwise wrote the file.
        if label_map is not None:
            uniq = np.unique(label_map)
            n_grains = int((uniq != 0).sum()) if 0 in uniq else int(uniq.size)
            if min_grains is not None and n_grains < int(min_grains):
                raise ValueError(
                    f"too_few_grains: n_grains={n_grains} < min_grains={int(min_grains)}"
                )
        else:
            n_grains = -1

        if label_map is not None:
            decode_image_pixelwise(
                image_path=png_path,
                mean_q=mean_q,
                output_path=output_dream3d,
                label_map=label_map,
                spacing=spacing,
            )
        else:
            decode_image_pixelwise(
                image_path=png_path,
                mean_q=mean_q,
                output_path=output_dream3d,
                spacing=spacing,
            )
            # Recover grain count from the file we just wrote so the
            # min_grains floor still applies to the default segmenter path.
            try:
                import h5py
                with h5py.File(output_dream3d, "r") as _f:
                    found = [None]
                    def _cb(name, obj):
                        if name.endswith("Grain Data/EulerAngles") and found[0] is None:
                            found[0] = int(obj.shape[0])
                    _f.visititems(_cb)
                n_grains = int(found[0]) if found[0] else -1
            except Exception:
                n_grains = -1
            if min_grains is not None and 0 < n_grains < int(min_grains):
                raise ValueError(
                    f"too_few_grains: n_grains={n_grains} < min_grains={int(min_grains)}"
                )

    # ------------------------------------------------------------------
    # Microstructure visualization: render the polycrystal that DAMASK
    # will actually simulate (FeatureIds painted with hashed RGB colors,
    # plus the EulerAngles map). Saved alongside the dream3d so each sim
    # subdir has a quick visual record of the input geometry.
    # ------------------------------------------------------------------
    try:
        _render_dream3d_preview(
            output_dream3d, output_dream3d.parent / "microstructure.png"
        )
    except Exception as _exc:  # non-fatal: the simulation can still proceed
        print(f"[codec] WARN microstructure preview failed: {_exc!r}")

    return output_dream3d, int(n_grains)


def _render_dream3d_preview(dream3d_path: Path, out_png: Path) -> None:
    """Render a 2-panel PNG (FeatureIds | per-grain orientation) of the polycrystal.

    Left panel  : grain IDs hashed to deterministic random RGB.
    Right panel : per-grain orientation coloring — for each FeatureId, the
    mean Bunge Euler over its voxels is computed (via the dominant
    eigenvector of the quaternion outer-product matrix) and broadcast back
    to all voxels of that grain. This matches what DAMASK actually
    simulates: the dream3d -> .vti + material.yaml conversion collapses
    voxels by FeatureId and assigns ONE orientation per grain. The
    per-pixel EulerAngles in the dream3d are a diagnostic write only.
    """
    import h5py
    from PIL import Image as _PILImage

    with h5py.File(dream3d_path, "r") as f:
        fid_obj: list = [None]
        eul_obj: list = [None]

        def _cb(name, obj):
            if name.endswith("CellData/FeatureIds") and fid_obj[0] is None:
                fid_obj[0] = obj[()]
            elif name.endswith("CellData/EulerAngles") and eul_obj[0] is None:
                eul_obj[0] = obj[()]

        f.visititems(_cb)

    if fid_obj[0] is None:
        return  # nothing to render

    fid = np.asarray(fid_obj[0]).squeeze()  # -> (H, W)
    if fid.ndim != 2:
        return

    # Hash grain IDs to deterministic RGB colors
    ids = fid.astype(np.int64)
    rng = np.random.default_rng(seed=0)
    max_id = int(ids.max()) + 1
    palette = rng.integers(40, 240, size=(max_id, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)  # background
    grain_rgb = palette[ids]  # (H, W, 3)

    # Per-grain orientation panel: collapse per-pixel Eulers by FeatureId
    if eul_obj[0] is not None:
        eul = np.asarray(eul_obj[0]).squeeze()  # (H, W, 3) radians
    else:
        eul = None

    if eul is not None and eul.ndim == 3 and eul.shape[-1] == 3:
        # Bunge ZXZ Euler -> unit quaternion (w, x, y, z), per pixel.
        p1 = eul[..., 0] * 0.5
        P_ = eul[..., 1] * 0.5
        p2 = eul[..., 2] * 0.5
        cP, sP = np.cos(P_), np.sin(P_)
        cm, sm = np.cos(p1 - p2), np.sin(p1 - p2)
        cp, sp = np.cos(p1 + p2), np.sin(p1 + p2)
        q = np.stack([cP * cp, sP * cm, sP * sm, cP * sp], axis=-1)
        q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
        q_flat = q.reshape(-1, 4)
        ids_flat = ids.reshape(-1)

        # Mean quaternion per grain via dominant eigenvector of sum(q q^T).
        # (Standard Markley-style quaternion averaging.)
        q_mean_per_id = np.zeros((max_id, 4), dtype=np.float64)
        for g in np.unique(ids_flat):
            if g <= 0:
                continue
            mask = ids_flat == g
            qg = q_flat[mask].astype(np.float64)
            M = qg.T @ qg
            _, vecs = np.linalg.eigh(M)
            qm = vecs[:, -1]
            if qm[0] < 0.0:
                qm = -qm
            q_mean_per_id[g] = qm

        # Quaternion -> Bunge Euler -> RGB (same wrap-to-[0,1] convention as
        # the previous per-pixel preview, so visualisations remain comparable
        # across runs; the only change is that all voxels of one grain now
        # share one colour, matching what DAMASK actually consumes).
        qm = q_mean_per_id
        w, x, y, z = qm[:, 0], qm[:, 1], qm[:, 2], qm[:, 3]
        Phi = np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
        sinPhi = np.sin(Phi)
        small = sinPhi < 1e-6
        phi1 = np.where(
            small,
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
            np.arctan2(2.0 * (x * z + w * y), 2.0 * (w * x - y * z)),
        )
        phi2 = np.where(
            small,
            np.zeros_like(phi1),
            np.arctan2(2.0 * (x * z - w * y), 2.0 * (w * x + y * z)),
        )
        eul_per_id = np.stack([phi1, Phi, phi2], axis=-1)
        eul_norm_id = (eul_per_id / (2.0 * np.pi)) % 1.0
        rgb_per_id = (eul_norm_id * 255.0).astype(np.uint8)
        rgb_per_id[0] = (0, 0, 0)  # background id stays black
        eul_rgb = rgb_per_id[ids]  # (H, W, 3)
    else:
        eul_rgb = np.zeros_like(grain_rgb)

    # Side-by-side panel with a 4-px white separator
    H, W = grain_rgb.shape[:2]
    sep = np.full((H, 4, 3), 255, dtype=np.uint8)
    panel = np.concatenate([grain_rgb, sep, eul_rgb], axis=1)
    _PILImage.fromarray(panel).save(out_png)


def save_metadata(meta: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(meta, indent=2, default=str))
