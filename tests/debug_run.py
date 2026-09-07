"""Debug entry point for microsam_segmentation_task.

Set breakpoints anywhere in the source, then run the
"Debug: microsam_segmentation_task" launch configuration in VS Code.
"""

from ngio import ChannelSelectionModel

from fractal_microsam_segmentation_task.microsam_segmentation_task import (
    microsam_segmentation_task,
)

print("Start debug run")  # Debugging output

microsam_segmentation_task(
    zarr_url="C:/Repos/test_data/test_microsam/image_01.zarr",
    channel=ChannelSelectionModel(identifier="channel_0", mode="label"),
)
