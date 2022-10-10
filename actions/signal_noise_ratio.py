from pathlib import Path
from typing import List

import numpy as np
import os

from util import (
    DicomSeriesList,
    param_or_default,
    acquisition_by_series_description,
    image_data_by_series_description,
    retrieve_ellipse_parameters,
    get_pixel_spacing,
    get_pixel_values_circle,
    get_background_rois_pixel_values,
    plot_rectangles_and_circles_on_image,
)


def signal_noise_ratio(parsed_input: List[DicomSeriesList], result, action_config):
    """
    Method for determining the signal to noise ratio as specified by the NVKF
    https://nvkf.nl/resources/LeidraadKwaliteitscontroleRadiologischeApparatuur3.0.pdf @ 203

    SNR = 0.655 * Sphantom / SDbackground

    Valid iff the mean / standard deviation of the background roi is 1.91

    """
    default_signal_roi_diameter_mm = 150
    default_noise_roi_sides_mm = 7
    default_background_roi_shift_mm = 108

    signal_roi_diameter_mm = float(param_or_default(
        action_config["params"],
        "snr_signal_roi_diameter_mm",
        default_signal_roi_diameter_mm,
    ))
    noise_roi_sides_mm = float(param_or_default(
        action_config["params"], "snr_noise_roi_sides_mm", default_noise_roi_sides_mm
    ))
    background_roi_shift_mm = float(param_or_default(
        action_config["params"],
        "snr_background_roi_shift_mm",
        default_background_roi_shift_mm,
    ))

    series_description = action_config["params"]["snr_series_description"]
    acquisition = acquisition_by_series_description(series_description, parsed_input)
    image_data = image_data_by_series_description(series_description, parsed_input)

    actionName = "signal_noise_ratio"
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

    # determine center coordinates of phantom
    [center_x, center_y, _, _, _] = retrieve_ellipse_parameters(
        image_data, mask_air_bubble=True
    )

    # retrieve pixel spacing to get the pixel values from the correct positions
    pixel_spacing = get_pixel_spacing(acquisition)

    # determine mean pixel value of center roi
    center_roi_pixel_values = get_pixel_values_circle(
        image_data, center_x, center_y, signal_roi_diameter_mm, pixel_spacing
    )
    center_roi_mean = center_roi_pixel_values.mean()

    # determine standard deviation of background rois
    background_rois_values, background_rois = get_background_rois_pixel_values(
        image_data,
        center_x,
        center_y,
        sides_mm=noise_roi_sides_mm,
        shift_mm=background_roi_shift_mm,
        pixel_spacing=pixel_spacing,
    )
    background_roi_std = np.std(background_rois_values)

    # determine SNR
    signal_to_noise_ratio = 0.0
    if background_roi_std > 0.0001:
        signal_to_noise_ratio = 0.655 * (center_roi_mean / background_roi_std)

    # save result objects to directory relative to working directory
    snr_result_image_filename = str(
        Path(
            result.getDirName()
            + os.sep
            + "result_objects"
            + os.sep
            + "snr_rois_result.png"
        ).absolute()
    )  # gets instantiated relative to working directory
    snr_result_image_filename_path = os.path.dirname(snr_result_image_filename)

    print("Write snr_rois_result.png to: '" + snr_result_image_filename_path + "'")

    if not os.path.exists(snr_result_image_filename_path):
        os.makedirs(snr_result_image_filename_path)

    plot_rectangles_and_circles_on_image(
        image_data,
        rectangles=background_rois,
        circles=[[(center_x, center_y), signal_roi_diameter_mm / 2]],
        title="SNR",
        save_as=snr_result_image_filename,
    )

    result.addFloat("CombinedCoils", signal_to_noise_ratio)
    result.addObject("SNR rois", snr_result_image_filename)
