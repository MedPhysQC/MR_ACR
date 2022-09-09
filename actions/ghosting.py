from pathlib import Path
from typing import List

import numpy as np
import os

from util import (
    DicomSeriesList,
    acquisition_by_series_description,
    image_data_by_series_description,
    get_pixel_spacing,
    retrieve_ellipse_parameters,
    get_ghosting_rois_pixel_values,
    get_background_rois_pixel_values,
    get_pixel_values_circle,
    plot_rectangles_and_circles_on_image,
)


def ghosting(parsed_input: List[DicomSeriesList], result, action_config) -> None:
    """
    signal ghosting percentage = 100 * (abs(ghost_signal - noise_signal) / (2* mean_signal))
    """
    series_description = action_config["params"]["ghosting_series_description"]
    ghosting_roi_short_side_mm = action_config["params"]["ghosting_roi_short_side_mm"]
    ghosting_roi_long_side_mm = action_config["params"]["ghosting_roi_long_side_mm"]
    noise_roi_sides_mm = action_config["params"]["ghosting_noise_roi_sides_mm"]
    phantom_roi_mm = action_config["params"]["ghosting_phantom_roi_radius_mm"]
    roi_shift_mm = action_config["params"]["ghosting_background_roi_shift_mm"]

    acquisition = acquisition_by_series_description(series_description, parsed_input)
    image_data = image_data_by_series_description(series_description, parsed_input)

    # retrieve pixel spacing to get the pixel values from the correct positions, verify square pixels
    pixel_spacing = get_pixel_spacing(acquisition)

    [center_x, center_y, _, _, _] = retrieve_ellipse_parameters(
        image_data, mask_air_bubble=True
    )

    (
        ghosting_rois_values,
        ghosting_roi_definitions_for_plotting,
    ) = get_ghosting_rois_pixel_values(
        image_data,
        center_x,
        center_y,
        ghosting_roi_short_side_mm,
        ghosting_roi_long_side_mm,
        roi_shift_mm,
        pixel_spacing,
    )

    (
        noise_rois_values,
        noise_roi_definitions_for_plotting,
    ) = get_background_rois_pixel_values(
        image_data, center_x, center_y, noise_roi_sides_mm, roi_shift_mm, pixel_spacing
    )

    phantom_roi_values = get_pixel_values_circle(
        image_data, center_x, center_y, phantom_roi_mm, pixel_spacing
    )

    ghosting_rows_percentage = (
        100
        * (
            max(
                np.mean(ghosting_rois_values["left"]),
                np.mean(ghosting_rois_values["right"]),
            )
            - np.mean(noise_rois_values)
        )
        / (2.0 * np.mean(phantom_roi_values))
    )
    ghosting_columns_percentage = (
        100
        * (
            max(
                np.mean(ghosting_rois_values["top"]),
                np.mean(ghosting_rois_values["bottom"]),
            )
            - np.mean(noise_rois_values)
        )
        / (2.0 * np.mean(phantom_roi_values))
    )
    image_ghosting_result_filename = str(
        Path(result.getDirName() + os.sep + "result_objects" + os.sep + "image_ghosting_result.png").absolute()
    )  # gets instantiated relative to working directory

    image_ghosting_result_filename_path = os.path.dirname(
        image_ghosting_result_filename
    )
    if not os.path.exists(image_ghosting_result_filename_path):
        os.makedirs(image_ghosting_result_filename_path)

    plot_rectangles_and_circles_on_image(
        image_data,
        [*ghosting_roi_definitions_for_plotting, *noise_roi_definitions_for_plotting],
        [[(center_x, center_y), phantom_roi_mm / 2]],
        title="ghosting and noise rois",
        save_as=image_ghosting_result_filename,
    )
    result.addFloat("GhostingRowsPercentage", ghosting_rows_percentage)
    result.addFloat("GhostingColumnsPercentage", ghosting_columns_percentage)
