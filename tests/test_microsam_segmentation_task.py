"""Tests for microsam_segmentation_task.

microSAM model loading and inference are mocked via the ``mock_microsam``
fixture (defined in conftest.py) so these tests run without GPU or network
access.
"""

import pytest
from ngio import ChannelSelectionModel, open_ome_zarr_container

from fractal_microsam_segmentation_task.microsam_segmentation_task import (
    _format_label_name,
    microsam_segmentation_task,
)
from fractal_microsam_segmentation_task.utils import CreateMaskingRoiTable

# ---------------------------------------------------------------------------
# Unit tests – _format_label_name
# ---------------------------------------------------------------------------


class TestFormatLabelName:
    def test_placeholder_replaced(self):
        assert _format_label_name("{channel_identifier}_seg", "DAPI") == "DAPI_seg"

    def test_no_placeholder_unchanged(self):
        assert _format_label_name("my_label", "DAPI") == "my_label"

    def test_unknown_placeholder_raises(self):
        with pytest.raises(ValueError, match="channel_identifier"):
            _format_label_name("{wrong_key}_seg", "DAPI")


# ---------------------------------------------------------------------------
# Integration tests – microsam_segmentation_task (mocked microSAM)
# ---------------------------------------------------------------------------

# (shape, axes, channels, channel_to_select)
ZARR_CONFIGS = [
    ((1, 64, 64), ["c", "y", "x"], ["DAPI"], "DAPI"),
    ((2, 64, 64), ["c", "y", "x"], ["DAPI", "GFP"], "GFP"),
    ((1, 1, 64, 64), ["c", "z", "y", "x"], ["DAPI"], "DAPI"),
    ((2, 1, 64, 64), ["c", "z", "y", "x"], ["DAPI", "GFP"], "DAPI"),
]


@pytest.mark.parametrize("shape,axes,channels,channel_sel", ZARR_CONFIGS)
def test_label_is_created(
    mock_microsam, make_ome_zarr, shape, axes, channels, channel_sel
):
    """Task should write a label image for every supported axes configuration."""
    store = make_ome_zarr(shape=shape, axes=axes, channels=channels)
    label_name = "test_label"

    microsam_segmentation_task(
        zarr_url=str(store),
        channel=ChannelSelectionModel(identifier=channel_sel, mode="label"),
        label_name=label_name,
        overwrite=True,
    )

    container = open_ome_zarr_container(str(store))
    assert label_name in container.list_labels()


@pytest.mark.parametrize("shape,axes,channels,channel_sel", ZARR_CONFIGS)
def test_label_spatial_shape_matches_image(
    mock_microsam, make_ome_zarr, shape, axes, channels, channel_sel
):
    """Label spatial dimensions (yx) must match those of the source image."""
    store = make_ome_zarr(shape=shape, axes=axes, channels=channels)
    label_name = "shape_check"

    microsam_segmentation_task(
        zarr_url=str(store),
        channel=ChannelSelectionModel(identifier=channel_sel, mode="label"),
        label_name=label_name,
        overwrite=True,
    )

    container = open_ome_zarr_container(str(store))
    label = container.get_label(label_name)
    # last two dims are always y, x
    assert label.shape[-2:] == (shape[-2], shape[-1])


def test_label_name_channel_identifier_template(mock_microsam, make_ome_zarr):
    """The {channel_identifier} placeholder must be replaced by the channel label."""
    store = make_ome_zarr(
        shape=(2, 64, 64), axes=["c", "y", "x"], channels=["DAPI", "GFP"]
    )

    microsam_segmentation_task(
        zarr_url=str(store),
        channel=ChannelSelectionModel(identifier="GFP", mode="label"),
        label_name="{channel_identifier}_microsam",
        overwrite=True,
    )

    container = open_ome_zarr_container(str(store))
    assert "GFP_microsam" in container.list_labels()


def test_overwrite_existing_label(mock_microsam, make_ome_zarr):
    """Running the task twice with overwrite=True should not raise."""
    store = make_ome_zarr(shape=(1, 64, 64), axes=["c", "y", "x"], channels=["DAPI"])
    kwargs = dict(
        zarr_url=str(store),
        channel=ChannelSelectionModel(identifier="DAPI", mode="label"),
        label_name="overwrite_label",
        overwrite=True,
    )

    microsam_segmentation_task(**kwargs)
    # Second run must not raise
    microsam_segmentation_task(**kwargs)

    container = open_ome_zarr_container(str(store))
    assert "overwrite_label" in container.list_labels()


def test_masking_roi_table_created(mock_microsam, make_ome_zarr):
    """When CreateMaskingRoiTable is provided, the ROI table must be written."""
    store = make_ome_zarr(shape=(1, 64, 64), axes=["c", "y", "x"], channels=["DAPI"])
    label_name = "dapi_seg"
    table_name = f"{label_name}_masking_ROI_table"

    microsam_segmentation_task(
        zarr_url=str(store),
        channel=ChannelSelectionModel(identifier="DAPI", mode="label"),
        label_name=label_name,
        overwrite=True,
        create_masking_roi_table=CreateMaskingRoiTable(),
    )

    container = open_ome_zarr_container(str(store))
    assert label_name in container.list_labels()
    assert table_name in container.list_tables()


def test_masked_iteration_with_label(mock_microsam, make_ome_zarr):
    """Task should run when iterating within an existing label mask (nuclei_mask).

    create_synthetic_ome_zarr generates 'nuclei_mask' by default, so we can
    use it as the masking label without extra setup.
    """
    from fractal_tasks_utils.segmentation import IteratorConfig

    store = make_ome_zarr(shape=(1, 64, 64), axes=["c", "y", "x"], channels=["DAPI"])

    microsam_segmentation_task(
        zarr_url=str(store),
        channel=ChannelSelectionModel(identifier="DAPI", mode="label"),
        label_name="masked_seg",
        overwrite=True,
        iterator_configuration=IteratorConfig(
            masking_label_name="nuclei_mask",
        ),
    )

    container = open_ome_zarr_container(str(store))
    assert "masked_seg" in container.list_labels()
