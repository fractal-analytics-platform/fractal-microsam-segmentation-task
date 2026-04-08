"""Segmentation utils"""

import logging
from enum import Enum
from typing import Any, Optional

import numpy as np
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
)
from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

logger = logging.getLogger(__name__)


class MODEL_ENUM(Enum):
    """Enum for model selection in micro-SAM segmentation."""

    VIT_H = "vit_h"
    VIT_L = "vit_l"
    VIT_B = "vit_b"
    VIT_T = "vit_t"
    VIT_L_LM = "vit_l_lm"
    VIT_B_LM = "vit_b_lm"
    VIT_T_LM = "vit_t_lm"
    VIT_L_EM_ORGANELLES = "vit_l_em_organelles"
    VIT_B_EM_ORGANELLES = "vit_b_em_organelles"
    VIT_T_EM_ORGANELLES = "vit_t_em_organelles"
    VIT_B_MEDICAL_IMAGING = "vit_b_medical_imaging"
    VIT_H_HISTOPATHOLOGY = "vit_h_histopathology"
    VIT_L_HISTOPATHOLOGY = "vit_l_histopathology"
    VIT_B_HISTOPATHOLOGY = "vit_b_histopathology"


def load_model_with_decoder(
    model_type: str,
    device: str,
    model_path: Optional[str] = None,
) -> InstanceSegmentationWithDecoder:
    """Load an exported model with decoder module for segmentation.

    This uses micro-SAM's get_predictor_and_segmenter to load the
    segmentation which handles decoder-based (AIS) mode.

    Args:
        model_type: SAM model type (e.g., 'vit_b_lm', 'vit_l_lm')
        device: Device to load model on ('cuda' or 'cpu')
        model_path: Path to a custom model checkpoint (.pt file). If None, the
            pre-trained micro-SAM model for `model_type` is downloaded/used
            from cache.

    Returns:
        segmenter: InstanceSegmentationWithDecoder for generating masks

    Raises:
        Exception: If model loading fails
    """
    logger.info(f"Loading model with {model_type} segmentation")

    # Use get_predictor_and_segmenter which properly handles both modes.
    # When checkpoint=None, micro-SAM downloads/uses the cached pre-trained
    # model for model_type.
    _, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=model_path,
        device=device,
        segmentation_mode="ais",
    )

    if not isinstance(segmenter, InstanceSegmentationWithDecoder):
        raise TypeError(
            "Expected InstanceSegmentationWithDecoder for AIS mode, "
            f"got {type(segmenter).__name__}"
        )

    return segmenter


def segment_image(
    image: np.ndarray,
    segmenter: InstanceSegmentationWithDecoder,
    generate_kwargs: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Run instance segmentation on a single image.

    Args:
        image: Input image as 2D numpy array
        segmenter: SAM segmenter (InstanceSegmentationWithDecoder)
        generate_kwargs: Optional parameters for generate() method (decoder thresholds)

    Returns:
        Instance segmentation masks as 2D numpy array with integer labels
    """
    generate_kwargs = generate_kwargs or {}

    if not isinstance(segmenter, InstanceSegmentationWithDecoder):
        raise TypeError(
            "segmenter must be InstanceSegmentationWithDecoder for AIS segmentation"
        )

    # micro_sam interprets image.shape[-1] as channels on 3-D arrays, so
    # always pass a 2-D (H, W) array. Remember the original shape so we can
    # restore extra leading dimensions for the writer.
    extra_dims = image.shape[:-2]  # e.g. (1,) for (1, H, W), () for (H, W)
    image_2d = image.reshape(-1, image.shape[-2], image.shape[-1])[0]  # (H, W)

    print(f"Image shape for micro_sam: {image_2d.shape}, {generate_kwargs=}")
    segmenter.initialize(image_2d)
    # generate() returns a (H, W) label array with integer instance IDs
    labels_2d = segmenter.generate(**generate_kwargs)

    print(f"Generated {labels_2d.max()} instances, shape={labels_2d.shape}")

    # Restore leading dimensions so the ngio writer can squeeze them back out
    for _ in extra_dims:
        labels_2d = labels_2d[np.newaxis]

    return labels_2d.astype(np.uint32)
