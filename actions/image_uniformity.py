from pathlib import Path
from typing import List

import numpy as np
import os

from util import (
    plot_rectangles_and_circles_on_image,
    get_points_in_circle,
    get_pixel_spacing,
    retrieve_ellipse_parameters,
    image_data_by_series_description,
    acquisition_by_series_description,
    DicomSeriesList,
)


def image_uniformity(
    parsed_input: List[DicomSeriesList], result, action_config
) -> None:
    """
    percentual image uniformity = 100 * (1-((max_signal - min_signal) / (max_signal + min signal))
    """
    series_description = action_config["params"]["image_uniformity_series_description"]
    acquisition = acquisition_by_series_description(series_description, parsed_input)
    image_data = image_data_by_series_description(series_description, parsed_input)

    actionName = "image_uniformity"
    print(
        "action "
        + actionName
        + " called for "
        + "study_instance_uid: '"
        + acquisition[0]["StudyInstanceUID"].value
        + "' series_instance_uid: '"
        + acquisition[0]["SeriesInstanceUID"].value
        + "'"
    )

    [center_x, center_y, _, _, _] = retrieve_ellipse_parameters(
        image_data, mask_air_bubble=True
    )

    pixel_spacing = get_pixel_spacing(acquisition)

    phantom_roi_size = float(action_config["params"]["image_uniformity_phantom_roi_size_mm"])
    uniformity_roi_size = float(action_config["params"][
        "image_uniformity_uniformity_roi_size_mm"
    ])
    # subtract the radius for the smaller roi to stay inside the phantom roi
    pixel_radius_uniformity = uniformity_roi_size * pixel_spacing / 2
    pixel_radius_phantom = (
        phantom_roi_size * pixel_spacing / 2
    )  # - pixel_radius_uniformity

    # set variables to keep track of found values
    max_mean = -1e99
    max_mean_center = [0, 0]
    min_mean = 1e99
    min_mean_center = [0, 0]

    # determine all points to check for signal level, subtract pixel_radius_uniformity because that circle roi needs to
    # fall completely within the phantom roi
    centers_to_check = get_points_in_circle(
        radius=pixel_radius_phantom - pixel_radius_uniformity,
        x_center=center_x,
        y_center=center_y,
    )
    for center in centers_to_check:
        # determine all points within relative to its center
        sub_points = get_points_in_circle(
            radius=pixel_radius_uniformity, x_center=center[0], y_center=center[1]
        )
        mean = np.mean([image_data[y][x] for x, y in sub_points])
        # compare mean with min and max and if updated, set coordinates for center of roi
        if mean > max_mean:
            max_mean = mean
            max_mean_center = [center[0], center[1]]
        if mean < min_mean:
            min_mean = mean
            min_mean_center = [center[0], center[1]]

    # determine uniformity as specified by nvkf
    image_uniformity_percent = 100 * (1 - (max_mean - min_mean) / (max_mean + min_mean))

    # create data for plotting reasons
    circle_1 = [(max_mean_center[0], max_mean_center[1]), 7.5 / pixel_spacing]
    circle_2 = [(min_mean_center[0], min_mean_center[1]), 7.5 / pixel_spacing]

    image_uniformity_result_filename = str(
        Path(
            result.getDirName()
            + os.sep
            + "result_objects"
            + os.sep
            + "image_uniformity_result.png"
        ).absolute()
    )  # gets instantiated relative to working directory
    image_uniformity_result_filename_path = os.path.dirname(
        image_uniformity_result_filename
    )
    print("Write image_uniformity_result.png to: '" + image_uniformity_result_filename + "'")

    if not os.path.exists(image_uniformity_result_filename_path):
        os.makedirs(image_uniformity_result_filename_path)
    plot_rectangles_and_circles_on_image(
        image_data,
        [],
        [circle_1, circle_2, [(center_x, center_y), pixel_radius_phantom]],
        title="roi_1 and roi_2 for image uniformity",
        save_as=image_uniformity_result_filename,
    )
    result.addFloat("ImageUniformityPercentage", image_uniformity_percent)
    result.addObject("ImageUniformityPlot", image_uniformity_result_filename)
