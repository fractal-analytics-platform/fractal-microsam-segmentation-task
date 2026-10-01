"""Unit tests for utils_segmentation: automatic tiling and tile-seam merging.

These exercise segment_image's tile-shape auto-detection and the ported
_merge_tile_seam_splits logic directly, with a mocked SAM predictor/segmenter
(no real model I/O, no GPU/network access needed).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from micro_sam.instance_segmentation import TiledInstanceSegmentationWithDecoder

from fractal_microsam_segmentation_task.utils_segmentation import (
    _merge_tile_seam_splits,
    segment_image,
    select_device,
)


def _make_predictor_and_segmenter(
    generated_labels: np.ndarray, img_size: int
) -> tuple[MagicMock, MagicMock]:
    predictor = MagicMock()
    predictor.model.image_encoder.img_size = img_size
    segmenter = MagicMock(spec=TiledInstanceSegmentationWithDecoder)
    segmenter.generate.return_value = generated_labels
    return predictor, segmenter


class TestSegmentImageAutomaticTiling:
    def test_no_tiling_when_image_fits_native_resolution(self):
        """An image within the encoder's native resolution is a single 'tile'."""
        image = np.zeros((64, 64), dtype=np.uint16)
        predictor, segmenter = _make_predictor_and_segmenter(
            np.zeros((64, 64), dtype=np.uint32), img_size=1024
        )

        segment_image(image, predictor=predictor, segmenter=segmenter, halo=(128, 128))

        _, kwargs = segmenter.initialize.call_args
        assert kwargs["tile_shape"] == (64, 64)
        assert kwargs["halo"] == (128, 128)

    def test_tiles_only_the_axis_that_exceeds_native_resolution(self):
        """Only axes larger than img_size get clamped; the other keeps its full size."""
        image = np.zeros((250, 80), dtype=np.uint16)
        predictor, segmenter = _make_predictor_and_segmenter(
            np.zeros((250, 80), dtype=np.uint32), img_size=100
        )

        result = segment_image(
            image, predictor=predictor, segmenter=segmenter, halo=(10, 10)
        )

        _, kwargs = segmenter.initialize.call_args
        assert kwargs["tile_shape"] == (100, 80)
        assert result.shape == (250, 80)

    def test_seam_merge_only_invoked_when_tiling_is_triggered(self, monkeypatch):
        merge_calls = []
        monkeypatch.setattr(
            "fractal_microsam_segmentation_task.utils_segmentation._merge_tile_seam_splits",
            lambda masks, *a, **k: merge_calls.append((a, k)) or masks,
        )

        small_image = np.zeros((64, 64), dtype=np.uint16)
        predictor, segmenter = _make_predictor_and_segmenter(
            np.zeros((64, 64), dtype=np.uint32), img_size=1024
        )
        segment_image(small_image, predictor=predictor, segmenter=segmenter)
        assert merge_calls == []

        large_image = np.zeros((250, 80), dtype=np.uint16)
        predictor, segmenter = _make_predictor_and_segmenter(
            np.zeros((250, 80), dtype=np.uint32), img_size=100
        )
        segment_image(large_image, predictor=predictor, segmenter=segmenter)
        assert len(merge_calls) == 1

    def test_rejects_non_tiled_segmenter(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        predictor = MagicMock()
        predictor.model.image_encoder.img_size = 1024
        with pytest.raises(TypeError, match="TiledInstanceSegmentationWithDecoder"):
            segment_image(image, predictor=predictor, segmenter=MagicMock())


class TestMergeTileSeamSplits:
    def test_straight_seam_aligned_split_is_merged(self):
        """A single object cut by a straight, seam-aligned tile boundary is rejoined."""
        masks = np.zeros((100, 100), dtype=np.int32)
        masks[40:50, 20:80] = 1  # top half of the split object
        masks[50:60, 20:80] = 2  # bottom half of the split object
        masks[10:20, 10:20] = 3  # unrelated object, must stay untouched

        merged = _merge_tile_seam_splits(masks, tile_shape=(50, 100), halo=(10, 10))

        merged_labels = set(np.unique(merged)) - {0}
        assert len(merged_labels) == 2  # the split pair collapsed into one label
        assert (merged[40:50, 20:80] == merged[50:60, 20:80]).all()
        assert (merged[10:20, 10:20] == 3).all()

    def test_curved_touching_boundary_is_not_merged(self):
        """Two distinct cells that happen to touch near a seam along a curved boundary
        must not be merged, even though they form a long run of the same label pair."""
        masks = np.zeros((100, 100), dtype=np.int32)
        for col in range(20, 80):
            # Diagonal boundary sweeping across a 20px band: not a straight,
            # seam-aligned line, but a plausible organic cell-cell contact.
            boundary_row = 40 + (col - 20) // 3
            masks[34:boundary_row, col] = 1
            masks[boundary_row:66, col] = 2

        merged = _merge_tile_seam_splits(masks, tile_shape=(50, 100), halo=(10, 10))

        assert np.array_equal(merged, masks)

    def test_no_objects_returns_input_unchanged(self):
        masks = np.zeros((100, 100), dtype=np.int32)
        merged = _merge_tile_seam_splits(masks, tile_shape=(50, 100), halo=(10, 10))
        assert merged is masks


class TestSelectDevice:
    def test_returns_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "fake-gpu")
        assert select_device() == "cuda"

    def test_raises_without_cuda_by_default(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            select_device()

    def test_falls_back_to_cpu_when_allowed(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert select_device(allow_cpu=True) == "cpu"
