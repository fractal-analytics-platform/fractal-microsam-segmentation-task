"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Niklas Khoss & Joel Luethi"


DOCS_LINK = (
    "https://github.com/fractal-analytics-platform/fractal-microsam-segmentation-task"
)


TASK_LIST = [
    ParallelTask(
        name="microSAM Segmentation",
        executable="microsam_segmentation_task.py",
        # Modify the meta according to your task requirements
        # If the task requires a GPU, add "needs_gpu": True
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Instance Segmentation", "Classical segmentation"],
        docs_info="file:docs_info/microsam_segmentation_task.md",
    ),
]
