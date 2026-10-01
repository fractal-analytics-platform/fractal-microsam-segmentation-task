"""Segmentation utils"""

import contextlib
import io
import logging
import os
import random
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
)
from micro_sam.instance_segmentation import TiledInstanceSegmentationWithDecoder
from segment_anything.predictor import SamPredictor

logger = logging.getLogger(__name__)


class MODEL_ENUM(Enum):
    """Raw micro-SAM model identifiers, as passed to get_predictor_and_segmenter.

    The Tiny (vit_t, vit_t_lm, vit_t_em_organelles) models are deliberately absent, see
    the note on MODEL_TYPE.
    """

    VIT_H = "vit_h"
    VIT_L = "vit_l"
    VIT_B = "vit_b"
    VIT_L_LM = "vit_l_lm"
    VIT_B_LM = "vit_b_lm"
    VIT_L_EM_ORGANELLES = "vit_l_em_organelles"
    VIT_B_EM_ORGANELLES = "vit_b_em_organelles"
    VIT_B_MEDICAL_IMAGING = "vit_b_medical_imaging"
    VIT_H_HISTOPATHOLOGY = "vit_h_histopathology"
    VIT_L_HISTOPATHOLOGY = "vit_l_histopathology"
    VIT_B_HISTOPATHOLOGY = "vit_b_histopathology"


# TODO: once a cluster-hosted model zoo exists, replace MODEL_TYPE/MODEL_ENUM
# (both static enums) with a dynamically built list so new checkpoints show up
# in the Fractal dropdown without a code change/release here.
class MODEL_TYPE(Enum):
    """Model choices shown in the Fractal dashboard dropdown.

    Friendly labels for the underlying micro-SAM checkpoints (MODEL_ENUM). vit_h/l/b/t
    are the original, non-microscopy-specific Segment Anything weights; all other entries
    are micro-SAM checkpoints fine-tuned for a specific imaging domain.
    Naming: Basic/Large/Huge refer to the underlying ViT encoder size (vit_b/l/h), not
    segmentation quality.

    The Tiny models (vit_t*, based on MobileSAM) are intentionally not offered: they need
    the `mobile_sam` package, which is neither on PyPI nor installable from conda-forge
    without pulling in conda pytorch. Fractal can only install this task's CUDA pytorch
    from PyPI wheels (conda CUDA builds need a `__cuda` virtual package, absent on the
    Fractal server), so Tiny support would require a git-sourced dependency. To bring it
    back: add `mobile-sam` (https://github.com/ChaoningZhang/MobileSAM, pinned to a
    commit) to the pixi pypi-dependencies, then restore the three enum entries here.
    A custom_model exported from a Tiny architecture fails to load for the same reason.
    """

    GENERIC_BASIC = "Generic - natural images (Basic)"
    GENERIC_LARGE = "Generic - natural images (Large)"
    GENERIC_HUGE = "Generic - natural images (Huge)"
    LIGHT_MICROSCOPY_BASIC = "Light Microscopy (Basic, default)"
    LIGHT_MICROSCOPY_LARGE = "Light Microscopy (Large)"
    ELECTRON_MICROSCOPY_ORGANELLES_BASIC = "Electron Microscopy - Organelles (Basic)"
    ELECTRON_MICROSCOPY_ORGANELLES_LARGE = "Electron Microscopy - Organelles (Large)"
    MEDICAL_IMAGING_BASIC = "Medical Imaging (Basic)"
    HISTOPATHOLOGY_BASIC = "Histopathology (Basic)"
    HISTOPATHOLOGY_LARGE = "Histopathology (Large)"
    HISTOPATHOLOGY_HUGE = "Histopathology (Huge)"


# Keep in sync with MODEL_TYPE above; every member must be mapped here.
MODEL_TYPE_TO_MODEL_ENUM: dict[MODEL_TYPE, MODEL_ENUM] = {
    MODEL_TYPE.GENERIC_BASIC: MODEL_ENUM.VIT_B,
    MODEL_TYPE.GENERIC_LARGE: MODEL_ENUM.VIT_L,
    MODEL_TYPE.GENERIC_HUGE: MODEL_ENUM.VIT_H,
    MODEL_TYPE.LIGHT_MICROSCOPY_BASIC: MODEL_ENUM.VIT_B_LM,
    MODEL_TYPE.LIGHT_MICROSCOPY_LARGE: MODEL_ENUM.VIT_L_LM,
    MODEL_TYPE.ELECTRON_MICROSCOPY_ORGANELLES_BASIC: MODEL_ENUM.VIT_B_EM_ORGANELLES,
    MODEL_TYPE.ELECTRON_MICROSCOPY_ORGANELLES_LARGE: MODEL_ENUM.VIT_L_EM_ORGANELLES,
    MODEL_TYPE.MEDICAL_IMAGING_BASIC: MODEL_ENUM.VIT_B_MEDICAL_IMAGING,
    MODEL_TYPE.HISTOPATHOLOGY_BASIC: MODEL_ENUM.VIT_B_HISTOPATHOLOGY,
    MODEL_TYPE.HISTOPATHOLOGY_LARGE: MODEL_ENUM.VIT_L_HISTOPATHOLOGY,
    MODEL_TYPE.HISTOPATHOLOGY_HUGE: MODEL_ENUM.VIT_H_HISTOPATHOLOGY,
}


def log_compute_environment() -> None:
    """Log torch/CUDA build and GPU visibility, to diagnose silent CPU fallbacks."""
    logger.info(
        f"torch={torch.__version__}, torch CUDA build={torch.version.cuda}, "
        f"cuda available={torch.cuda.is_available()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')}"
    )
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")


def _is_architecture_supported(
    capability: tuple[int, int], compiled_architectures: list[str]
) -> bool:
    """Check whether a torch build has kernels for a GPU compute capability.

    Args:
        capability: (major, minor) compute capability of the GPU, e.g. (7, 0) for V100.
        compiled_architectures: `torch.cuda.get_arch_list()`, e.g. ['sm_75',
            'compute_90'].

    Returns:
        True if there are SASS kernels for the exact capability, or PTX for an equal or
        lower one (which the driver can JIT-compile).
    """
    device_cc = capability[0] * 10 + capability[1]
    for architecture in compiled_architectures:
        kind, _, number = architecture.partition("_")
        if not number.isdigit():
            continue
        if kind == "sm" and int(number) == device_cc:
            return True
        if kind == "compute" and int(number) <= device_cc:
            return True
    return False


def select_device(allow_cpu: bool = False) -> str:
    """Select the inference device and log the compute environment.

    Fails fast when CUDA is unavailable or the torch build has no kernels for the GPU.
    micro-SAM ViT inference on CPU is roughly an order of magnitude slower and is almost
    always an environment mistake (e.g. a CPU-only torch build) rather than intended.

    Args:
        allow_cpu: If True, fall back to CPU (with a warning) instead of raising.

    Returns:
        'cuda' or 'cpu'.

    Raises:
        RuntimeError: If CUDA is unavailable and `allow_cpu` is False.
    """
    log_compute_environment()
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        compiled_architectures = torch.cuda.get_arch_list()
        if not _is_architecture_supported(capability, compiled_architectures):
            raise RuntimeError(
                f"This torch build has no kernels for the GPU's compute capability "
                f"{capability[0]}.{capability[1]} "
                f"(built for {compiled_architectures}). "
                "Use a torch wheel that still supports this GPU generation."
            )
        return "cuda"
    if not allow_cpu:
        raise RuntimeError(
            "CUDA is not available (see the logged torch build and CUDA_VISIBLE_DEVICES). "
            "Refusing to run micro-SAM on CPU; set allow_cpu=True to override."
        )
    logger.warning("CUDA is not available, running micro-SAM on CPU (slow).")
    return "cpu"


def _load_with_retry(
    loader: Callable[[], tuple[SamPredictor, TiledInstanceSegmentationWithDecoder]],
    description: str,
    max_attempts: int = 10,
) -> tuple[SamPredictor, TiledInstanceSegmentationWithDecoder]:
    """Load a micro-SAM model with retry logic for cluster safety.

    Multiple parallel workers may attempt to load or download the same
    checkpoint simultaneously, which can produce stale file handle errors on
    network-mounted filesystems. Retries with random backoff to recover.

    Args:
        loader: Zero-argument callable that returns the loaded (predictor, segmenter).
        description: Human-readable description used in log/error messages.
        max_attempts: Maximum number of attempts before raising.

    Returns:
        Loaded (predictor, segmenter) pair.

    Raises:
        RuntimeError: If the model cannot be loaded after max_attempts.
    """
    result = None
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = loader()
            if result is not None:
                break
        except Exception as e:
            last_error = e
            logger.warning(
                f"Failed to load {description} (attempt {attempt}/{max_attempts}): "
                f"{e!r}. Retrying..."
            )
            time.sleep(random.uniform(2, 7))

    if result is None:
        raise RuntimeError(
            f"Could not load {description} after {max_attempts} attempts."
        ) from last_error
    return result


def load_model_with_decoder(
    model_type: str,
    device: str,
    model_path: Optional[str] = None,
) -> tuple[SamPredictor, TiledInstanceSegmentationWithDecoder]:
    """Load an exported model with decoder module for segmentation.

    This uses micro-SAM's get_predictor_and_segmenter to load the
    segmentation which handles decoder-based (AIS) mode. The segmenter is
    always built with tiling support (`is_tiled=True`): `segment_image` picks
    a tile shape per image, tiling only when the image exceeds the encoder's
    native input resolution, so this has no effect on images that fit within it.

    Args:
        model_type: SAM model type (e.g., 'vit_b_lm', 'vit_l_lm'). If
            `model_path` is set and its checkpoint was exported for a
            different architecture, micro-SAM detects the mismatch from the
            checkpoint itself and overrides `model_type` accordingly (logged
            as a warning here rather than left as an easy-to-miss print).
        device: Device to load model on ('cuda' or 'cpu')
        model_path: Path to a custom model checkpoint (.pt file). If None, the
            pre-trained micro-SAM model for `model_type` is downloaded/used
            from cache.

    Returns:
        Tuple of (predictor, segmenter), where segmenter is a
        TiledInstanceSegmentationWithDecoder.

    Raises:
        RuntimeError: If model loading fails after retries.
    """
    logger.info(f"Loading model: {model_type}")

    description = f"micro-SAM model '{model_type}'"
    if model_path:
        description += f" from {model_path}"

    def _load() -> tuple[SamPredictor, TiledInstanceSegmentationWithDecoder]:
        # When checkpoint=None, micro-SAM downloads/uses the cached pre-trained
        # model for model_type. Custom checkpoints must be lean exports
        # (model_state + decoder_state only, no training-time objects) — see
        # sam_trainer's export step; the fractal task does not special-case
        # legacy training-checkpoint-shaped files.
        # micro-SAM reports a model_type/checkpoint architecture mismatch via a
        # bare print(), which is easy to miss in Fractal's logs. Capture it and
        # re-emit through our own logger instead of relying on the raw print.
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model_type,
                checkpoint=model_path,
                device=device,
                segmentation_mode="ais",
                is_tiled=True,
            )
        warning_message = captured_stdout.getvalue().strip()
        if warning_message:
            logger.warning(f"micro-SAM model loading warning: {warning_message}")
        if not isinstance(segmenter, TiledInstanceSegmentationWithDecoder):
            raise TypeError(
                "Expected TiledInstanceSegmentationWithDecoder for AIS mode, "
                f"got {type(segmenter).__name__}"
            )
        return predictor, segmenter

    return _load_with_retry(_load, description)


def _seam_transition(
    strip: np.ndarray, center: float
) -> Optional[tuple[int, int, float]]:
    """Find the label transition nearest the seam along a cross-seam strip.

    Returns `(label_before, label_after, offset)` if `strip` (a 1D cross-section
    perpendicular to the seam) contains exactly two distinct nonzero labels, else `None`.
    `center` is the strip index corresponding to the nominal seam pixel. With a wide search
    band (large `margin`), a strip can contain small far-away clusters of either label near
    the band's edges that are irrelevant to the seam itself (e.g. the same two cells also
    happen to be near each other well away from where they actually touch) — picking *the*
    transition among all label changes in the strip that's nearest `center`, rather than using
    each label's global extremal pixel, keeps the estimate meaningful regardless of how wide
    the band is.
    """
    nz_idx = np.flatnonzero(strip)
    if nz_idx.size == 0:
        return None
    values = strip[nz_idx]
    if np.unique(values).size != 2:
        return None
    change_positions = np.flatnonzero(np.diff(values) != 0)
    if change_positions.size == 0:
        return None
    offsets = (nz_idx[change_positions] + nz_idx[change_positions + 1]) / 2.0
    best = change_positions[np.argmin(np.abs(offsets - center))]
    offset = float((nz_idx[best] + nz_idx[best + 1]) / 2.0)
    return int(values[best]), int(values[best + 1]), offset


def _merge_along_seam_band(
    band: np.ndarray,
    center: float,
    min_run: int,
    max_offset_spread: int,
    union: Callable[[int, int], None],
) -> None:
    """Union label pairs from long, *straight* runs of a clean 2-label transition.

    `band` is a strip of pixels straddling a tile seam, shape (band_width, n_positions) —
    each column `i` is the cross-section of pixel values at position `i` along the seam,
    spanning a small margin on both sides of it. Two requirements distinguish a genuine
    tiling-seam split from two cells that merely happen to touch near a seam:
    1. Long run (>= min_run) of the *same* label pair.
    2. Most of the run sits at a nearly constant offset from the seam (straight, parallel to
       the seam — within `max_offset_spread` of the run's median offset), not wandering the
       way an organic cell-cell boundary does. Without this check, a long curved genuine
       contact between two distinct cells can satisfy "same pair for a while" and get
       incorrectly merged. A median-based "core" count rather than raw max-min is used
       deliberately: with a wide search band, `_seam_transition` can pick an unrelated,
       far-away transition for a handful of columns where the true near-seam transition is
       momentarily ambiguous, which makes one genuine, otherwise dead-straight run look far
       more scattered using raw max-min. Requiring most (not all) of the run to be tight
       tolerates that without accepting a genuinely curved boundary, which won't have a
       majority cluster near any single offset.
    """
    n_positions = band.shape[1]
    transitions = [_seam_transition(band[:, i], center) for i in range(n_positions)]
    pairs = [(t[0], t[1]) if t else None for t in transitions]

    def close_run(
        run_start: int, run_end: int, pair: Optional[tuple[int, int]]
    ) -> None:
        if pair is None or run_end - run_start < min_run:
            return
        offsets = np.array([transitions[i][2] for i in range(run_start, run_end)])
        core = np.abs(offsets - np.median(offsets)) <= max_offset_spread
        # Both an absolute floor (avoids a short run passing on a tiny, easy-to-hit majority)
        # and a high fraction (avoids a genuinely wandering boundary passing just because *some*
        # long sub-stretch happens to be locally flat) are required.
        if core.sum() >= min_run and core.mean() >= 0.85:
            union(*pair)

    run_start = 0
    run_pair = pairs[0] if n_positions else None
    for i in range(1, n_positions + 1):
        current = pairs[i] if i < n_positions else None
        if current == run_pair:
            continue
        close_run(run_start, i, run_pair)
        run_start = i
        run_pair = current


def _merge_tile_seam_splits(
    masks: np.ndarray,
    tile_shape: tuple[int, int],
    halo: tuple[int, int],
    margin: Optional[int] = None,
    min_run: int = 10,
    max_offset_spread: int = 8,
) -> np.ndarray:
    """Merge instances that tiled AIS inference split along tile-grid seams.

    `TiledInstanceSegmentationWithDecoder` stitches each tile's decoder output (foreground/
    center-distance/boundary-distance maps) with a hard cut at the tile's inner boundary —
    the halo only gives each tile's encoder extra context, predictions from adjacent tiles
    are never blended. A cell straddling a seam can get slightly different center/boundary
    predictions on each side, and the watershed-style instance decoding then treats that
    mismatch as a real boundary, splitting one cell into two near the seam line. The actual
    split boundary can land well off the exact seam coordinate (most cases within ~15px of
    the nominal seam, but a genuine straight split has been observed 58px away with
    halo=128). Since halo is exactly the size of the context window that can make a tile's
    prediction diverge before reaching the seam itself, the search band scales with it by
    default (`margin = halo`) rather than using a fixed constant.

    A pair is only merged if the transition between the two labels is long *and straight*
    (near-constant offset from the seam across the run) — a long but curved/wandering contact
    is what a genuine cell-cell touch near a seam looks like, and merging on "same pair for a
    while" alone both over-merges genuinely distinct touching cells and, via transitive
    union-find, can pull in an unrelated third instance that happens to share a label with one
    genuine split. The straightness check is what makes it safe to use a generous margin — a
    coincidentally long, dead-straight (within a few px) run between two unrelated cells over
    50+ px is not a realistic biological boundary shape, so widening the search window doesn't
    materially increase false merges.

    Tile seams are deterministic — the tiling grid always starts at (0, 0) (tiles are simple
    truncation, size `tile_shape` except a clipped last tile per axis), so seam lines fall at
    exact multiples of `tile_shape` regardless of halo.

    Args:
        masks: Instance segmentation labels from tiled AIS inference.
        tile_shape: The tile shape used for inference (same value passed to `initialize`).
        halo: The halo used for inference (same value passed to `initialize`). Used to size
            `margin` by default.
        margin: Half-width (pixels) of the band scanned on each side of a seam line for
            candidate split-label pairs. Defaults to `max(halo)` (at least 16) when not given.
        min_run: Minimum contiguous run length (pixels) of the same 2-label pair along a
            seam band to treat as a split rather than coincidental cell-to-cell proximity.
        max_offset_spread: Maximum allowed variation (pixels) in the transition's distance
            from the seam across a run — enforces that the split boundary is actually straight
            and seam-aligned, not an organic curved cell-cell contact.

    Returns:
        Masks with seam-split instances merged back into a single label (labels unchanged
        for anything not connected to a seam split).
    """
    if margin is None:
        margin = max(16, *halo)

    if masks.max() == 0:
        return masks

    parent = {int(label): int(label) for label in np.unique(masks) if label != 0}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    height, width = masks.shape
    for seam in range(tile_shape[0], height, tile_shape[0]):
        lo, hi = max(0, seam - margin), min(height, seam + margin)
        _merge_along_seam_band(
            masks[lo:hi, :], seam - lo, min_run, max_offset_spread, union
        )
    for seam in range(tile_shape[1], width, tile_shape[1]):
        lo, hi = max(0, seam - margin), min(width, seam + margin)
        _merge_along_seam_band(
            masks[:, lo:hi].T, seam - lo, min_run, max_offset_spread, union
        )

    roots = {label: find(label) for label in parent}
    if all(root == label for label, root in roots.items()):
        return masks

    merged = masks.copy()
    for label, root in roots.items():
        if root != label:
            merged[masks == label] = root
    return merged


def segment_image(
    image: np.ndarray,
    predictor: SamPredictor,
    segmenter: TiledInstanceSegmentationWithDecoder,
    halo: tuple[int, int] = (128, 128),
    generate_kwargs: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Run instance segmentation on a single image, tiling only if needed.

    Without tiling, SAM's encoder resizes the image so its longest side matches a fixed
    native resolution (`predictor.model.image_encoder.img_size`, typically 1024px) before
    segmentation. For images much larger than that, this silently shrinks objects far below
    the scale the decoder was trained on, producing sparse, misaligned masks. To keep images
    as large as possible without triggering that resize, this tiles only the axes that
    actually exceed the encoder's native resolution, using the largest tile size that avoids
    it (`min(native_size, image_dim)` per axis) — an image that already fits is processed as
    a single tile, identical to the untiled code path.

    Args:
        image: Input image as 2D numpy array
        predictor: SAM predictor returned alongside `segmenter` by `load_model_with_decoder`.
        segmenter: SAM segmenter (TiledInstanceSegmentationWithDecoder)
        halo: Overlap between tiles used to stitch tiled inference. Only used, and only
            applied to a `TiledInstanceSegmentationWithDecoder`, when the image is actually
            tiled (i.e. exceeds the encoder's native input resolution on some axis).
        generate_kwargs: Optional parameters for generate() method (decoder thresholds)

    Returns:
        Instance segmentation masks as 2D numpy array with integer labels
    """
    generate_kwargs = generate_kwargs or {}

    if not isinstance(segmenter, TiledInstanceSegmentationWithDecoder):
        raise TypeError(
            "segmenter must be TiledInstanceSegmentationWithDecoder for AIS segmentation"
        )

    # micro_sam interprets image.shape[-1] as channels on 3-D arrays, so
    # always pass a 2-D (H, W) array. Remember the original shape so we can
    # restore extra leading dimensions for the writer.
    extra_dims = image.shape[:-2]  # e.g. (1,) for (1, H, W), () for (H, W)
    image_2d = image.reshape(-1, image.shape[-2], image.shape[-1])[0]  # (H, W)

    native_size = predictor.model.image_encoder.img_size
    tile_shape = (
        min(native_size, image_2d.shape[0]),
        min(native_size, image_2d.shape[1]),
    )
    is_tiled = tile_shape != image_2d.shape
    logger.debug(
        f"Image shape for micro_sam: {image_2d.shape}, {tile_shape=}, {is_tiled=}, "
        f"{generate_kwargs=}"
    )

    segmenter.initialize(image_2d, tile_shape=tile_shape, halo=halo)
    # generate() returns a (H, W) label array with integer instance IDs
    labels_2d = segmenter.generate(**generate_kwargs)

    if is_tiled:
        labels_2d = _merge_tile_seam_splits(labels_2d, tile_shape, halo=halo)

    logger.info(f"Generated {labels_2d.max()} instances, shape={labels_2d.shape}")

    # Restore leading dimensions so the ngio writer can squeeze them back out
    for _ in extra_dims:
        labels_2d = labels_2d[np.newaxis]

    return labels_2d.astype(np.uint32)
