"""This is the Python module for my_task."""

import logging

from fractal_tasks_utils.segmentation import (
    IteratorConfig,
    compute_segmentation,
    setup_segmentation_iterator,
)
from fractal_tasks_utils.segmentation._transforms import SegmentationTransformConfig
from ngio import ChannelSelectionModel, open_ome_zarr_container
from pydantic import Field, validate_call

from fractal_microsam_segmentation_task.utils import (
    AnyCreateRoiTableModel,
    CreateMaskingRoiTable,
    SkipCreateMaskingRoiTable,
)
from fractal_microsam_segmentation_task.utils_segmentation import (
    MODEL_TYPE,
    MODEL_TYPE_TO_MODEL_ENUM,
    load_model_with_decoder,
    segment_image,
    select_device,
)

logger = logging.getLogger("microsam_segmentation_task")


# from ngio.images._image import _parse_channel_selection


# FIXME: Adapt to simpler channel picker (not a list, but would need to
# introduce skip button)
# def _skip_segmentation(
#         channel: ChannelSelectionModel,
#         ome_zarr: OmeZarrContainer
#     ) -> bool:
#     """Check wheter to skip the current task based on the channel configuration.

#     If the channel selection specified in the channels parameter is not
#     valid for the provided OME-Zarr image, this function checks the
#     skip_if_missing attribute of the channels configuration.
#     If skip_if_missing is True, the function returns True, indicating that the task
#     should be skipped. If skip_if_missing is False, a ValueError is raised.

#     Args:
#         channels (CellposeChannels): The channel selection configuration.
#         ome_zarr (OmeZarrContainer): The OME-Zarr container to check against.

#     Returns:
#         bool: True if the task should be skipped due to missing channels,
#         False otherwise.

#     """
#     image = ome_zarr.get_image()
#     try:
#         _parse_channel_selection(image=image, channel_selection=[channel])
#     except NgioValueError as e:
#         if channels.skip_if_missing:
#             logger.warning(
#                 f"Channel selection {channels_list} is not valid for the provided "
#                 "image, but skip_if_missing is set to True. Skipping segmentation."
#             )
#             logger.debug(f"Original error message: {e}")
#             return True
#         else:
#             raise ValueError(
#                 f"Channel selection {channels_list} is not valid for the provided "
#                 "image. If you want to skip processing when channels are missing, "
#                 "set skip_if_missing to True."
#             ) from e
#     return False


def _format_label_name(label_name_template: str, channel_identifier: str) -> str:
    """Format the label name based on the provided template and channel identifier.

    Args:
        label_name_template (str): The template for the label name. This
        might contain a placeholder "{channel_identifier}" which will be replaced
        by the channel identifier or no placeholder at all,
        in which case the channel identifier will be ignored.
        channel_identifier (str): The channel identifier to insert into the
            label name template.

    Returns:
        str: The formatted label name.
    """
    try:
        label_name = label_name_template.format(channel_identifier=channel_identifier)
    except KeyError as e:
        raise ValueError(
            "Label Name format error only allowed placeholder is "
            f"'channel_identifier'. {{{e}}} was provided."
        ) from e
    return label_name


@validate_call
def microsam_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channel: ChannelSelectionModel,
    label_name: str = "{channel_identifier}_microsam_segmented",
    level_path: str | None = None,
    # Iteration parameters
    model_type: MODEL_TYPE = MODEL_TYPE.LIGHT_MICROSCOPY_BASIC,
    custom_model: str | None = None,
    allow_cpu: bool = False,
    halo: tuple[int, int] = (128, 128),
    center_distance_threshold: float = 0.5,
    boundary_distance_threshold: float = 0.5,
    foreground_threshold: float = 0.5,
    iterator_configuration: IteratorConfig | None = None,
    pre_post_process: SegmentationTransformConfig = Field(  # noqa: B008
        default_factory=SegmentationTransformConfig
    ),
    create_masking_roi_table: AnyCreateRoiTableModel = Field(  # noqa: B008
        default_factory=SkipCreateMaskingRoiTable
    ),
    overwrite: bool = True,
) -> None:
    """Segment an image using a simple thresholding method.

    This taks demostrates how to use the SegmentationIterator and
    MaskedSegmentationIterator to perform segmentation. We provide a separate
    utility package with some helper functions and classes to streamline the development
    of measurement tasks. For more infos check:
    https://github.com/fractal-analytics-platform/fractal-microsam-segmentation-task/fractal-tasks-utils

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channel (ChannelSelectionModel): Select the input channel to be used for
            segmentation.
        label_name (str): Name of the resulting label image. Optionally, it can contain
            a placeholder "{channel_identifier}" which will be replaced by the
            first channel identifier specified in the channels parameter.
        level_path (str | None): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        model_type (MODEL_TYPE): Which pretrained micro-SAM model to use, grouped by
            imaging domain (light microscopy, electron microscopy, histopathology,
            medical imaging) and encoder size (Basic = fastest/least accurate, Huge =
            slowest/most accurate). Light Microscopy (Basic) is a good default for
            most fluorescence/brightfield data. Still required when custom_model is
            set (it selects the encoder architecture to build): if it does not match
            custom_model's actual architecture, micro-SAM detects this from the
            checkpoint itself and corrects it automatically, logging a warning.
        custom_model (str | None): Path to a custom SAM model checkpoint, used instead
            of the pretrained checkpoint for model_type.
        allow_cpu (bool): If False (default), the task fails when no GPU is available
            instead of silently running on CPU, which is far slower. Set to True only
            for small tests.
        halo (tuple[int, int]): Overlap (y, x) between tiles, in pixels. Only used
            when an image exceeds the model's native input resolution (typically
            1024px) and is therefore tiled to avoid downscaling objects; ignored for
            smaller images. Larger values reduce tile-seam segmentation artifacts at
            the cost of extra compute.
        center_distance_threshold (float): Only pixels the model predicts to lie close
            enough to an object's centre (below this value) can seed a new instance;
            combined with boundary_distance_threshold. Range 0-1 (default: 0.5).
            Lower: fewer, stricter seeds (risk of missing/merging faint or touching
            objects). Higher: more seeds (helps separate touching objects, but risks
            splitting a single object into several).
        boundary_distance_threshold (float): Only pixels predicted to lie far enough
            from an object's edge (below this value) can seed a new instance; combined
            with center_distance_threshold. Range 0-1 (default: 0.5).
            Lower: seeds must sit well inside an object, giving safer separation of
            touching objects but risking missed small/thin objects. Higher: seeds
            allowed closer to edges, recovering small objects but risking merged
            touching instances.
        foreground_threshold (float): Threshold on the predicted foreground
            probability that decides whether a pixel belongs to any object at all
            (independent of the two thresholds above). Range 0-1 (default: 0.5).
            Lower: more inclusive, recovers dim/faint objects but may pick up
            background noise as spurious objects. Higher: stricter, rejects noise but
            may lose faint or low-contrast objects.
            Tuning tip: touching objects merged into one label -> raise
            center/boundary_distance_threshold; one object split into fragments ->
            lower them; objects missing entirely -> lower foreground_threshold;
            spurious background objects -> raise foreground_threshold.
        iterator_configuration (IteratorConfiguration | None): Advanced
            configuration to control masked and ROI-based iteration.
        pre_post_process (SegmentationTransformConfig): Configuration for pre- and
            post-processing transforms applied by the iterator.
        create_masking_roi_table (AnyCreateRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    # Use the first of input_paths
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")
    # Validate that the specified channels are present in the image
    # FIXME: Add channel validation
    # if _skip_segmentation(channels=channel, ome_zarr=ome_zarr):
    #     return None

    # Format the label name based on the provided template and channel identifier
    label_name = _format_label_name(
        label_name_template=label_name, channel_identifier=channel.identifier
    )
    logger.info(f"Formatted label name: {label_name=}")

    # Based on model_type or custom_model
    predictor, segmenter = load_model_with_decoder(
        model_type=MODEL_TYPE_TO_MODEL_ENUM[model_type].value,
        device=select_device(allow_cpu=allow_cpu),
        model_path=custom_model,
    )

    # if advanced_parameters.verbose:
    #     logging.getLogger("cellpose").setLevel(logging.INFO)
    # else:
    #     logging.getLogger("cellpose").setLevel(logging.WARNING)

    microsam_kwargs = {
        "center_distance_threshold": center_distance_threshold,
        "boundary_distance_threshold": boundary_distance_threshold,
        "foreground_threshold": foreground_threshold,
    }

    # Set up the segmentation iterator
    iterator = setup_segmentation_iterator(
        zarr_url=zarr_url,
        channels=[channel],
        output_label_name=label_name,
        level_path=level_path,
        iterator_configuration=iterator_configuration,
        segmentation_transform_config=pre_post_process,
        overwrite=overwrite,
    )

    # Run the core segmentation loop
    compute_segmentation(
        segmentation_func=lambda x: segment_image(
            image=x,
            predictor=predictor,
            segmenter=segmenter,
            halo=halo,
            generate_kwargs=microsam_kwargs,
        ),
        iterator=iterator,
    )
    logger.info(f"label {label_name} successfully created at {zarr_url}")

    # Building a masking roi table
    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        label = ome_zarr.get_label(name=label_name, path=level_path)
        masking_roi_table = label.build_masking_roi_table()
        ome_zarr.add_table(
            name=table_name, table=masking_roi_table, overwrite=overwrite
        )
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=microsam_segmentation_task)
