"""Shared pytest fixtures for fractal-microsam-segmentation-task tests."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ngio import create_synthetic_ome_zarr


def _fake_segment_image(
    image: np.ndarray,
    predictor=None,
    segmenter=None,
    **kwargs,
) -> np.ndarray:
    """Mock segmentation: zeros with the input's shape, incl. leading dims.

    Mirrors segment_image's real contract of restoring leading dims (c, z, ...)
    collapsed for the 2-D microSAM call, which the ngio writer relies on.
    """
    extra_dims = image.shape[:-2]
    labels = np.zeros(image.shape[-2:], dtype=np.uint32)
    for _ in extra_dims:
        labels = labels[np.newaxis]
    return labels


@pytest.fixture
def mock_microsam(monkeypatch):
    """Patch load_model_with_decoder and segment_image to avoid real model I/O."""
    mock_predictor = MagicMock()
    mock_segmenter = MagicMock()
    monkeypatch.setattr(
        "fractal_microsam_segmentation_task.microsam_segmentation_task.load_model_with_decoder",
        MagicMock(return_value=(mock_predictor, mock_segmenter)),
    )
    monkeypatch.setattr(
        "fractal_microsam_segmentation_task.microsam_segmentation_task.segment_image",
        _fake_segment_image,
    )
    return mock_predictor, mock_segmenter


@pytest.fixture
def make_ome_zarr(tmp_path):
    """Factory fixture: create a small synthetic OME-Zarr container on demand.

    Usage::

        def test_something(make_ome_zarr):
            store = make_ome_zarr(
                shape=(2, 64, 64),
                axes=["c", "y", "x"],
                channels=["DAPI", "GFP"],
            )
    """

    def _factory(
        shape: tuple[int, ...],
        axes: list[str],
        channels: list[str],
        name: str | None = None,
    ) -> Path:
        store_name = name or ("_".join(axes) + ".zarr")
        store = tmp_path / store_name
        create_synthetic_ome_zarr(
            store=store,
            shape=shape,
            axes_names=axes,
            channels_meta=channels,
            overwrite=True,
        )
        return store

    return _factory
