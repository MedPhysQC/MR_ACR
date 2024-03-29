from enum import Enum
from typing import List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import os

import skimage.restoration


from util import (
    acquisition_by_series_description,
    DicomSeriesList,
    retrieve_ellipse_parameters,
    elliptical_mask,
)


class ShimDataTypes(Enum):
    WRAPPED_PHASE_RADIANS = "dPhi_rad"
    B0_PPM = "dB0_ppm"
    B0_HZ = "dB0_hz"


def shim(parsed_input: List[DicomSeriesList], result, action_config) -> None:
    """
    shim = 10e6 *  ((max(B0) - min(B0)) / mean(B0))
    Expects a phase and a magnitude image
    The magnitude image is used to determine the center position and phantom radius.
    This is used to create a mask with a radius of 95% of the phantom,
    leaving out the upper 20% as well to account for the air bubble in the top of the phantom
    """
    series_description = action_config["params"]["shim_series_description"]
    roi_radius_pct = float(action_config["params"]["shim_relative_roi_size"])
    top_margin = int(action_config["params"]["shim_top_margin"])

    # retrieve all slices
    acquisition = acquisition_by_series_description(series_description, parsed_input)

    actionName = "shim"
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

    # check if there are 4 images
    if len(acquisition) != 4:
        raise ValueError(f"SHIM: Expected 4 images, found {len(acquisition)}")

    # magnitude index = 0; used for determining center of the phantom
    magnitude = acquisition[0]

    # check if acquisition has FFE in 2001,1020
    private_2001_1020 = magnitude[0x20011020].value.strip()
    if private_2001_1020 != "FFE":
        raise ValueError(
            f"SHIM: Could not detect Philips double echo FFE. Expected 'FFE'; got {private_2001_1020}"
        )

    # phase1 index = 2, phase2 index=3
    phase_1 = acquisition[2]
    phase_2 = acquisition[3]

    # convert pixel values to radians
    if hasattr(phase_1, "RescaleSlope") and hasattr(phase_2, "RescaleSlope"):
        factor_1 = phase_1.RescaleSlope / 1000.0
        offset_1 = phase_1.RescaleIntercept / 1000.0
        factor_2 = phase_2.RescaleSlope / 1000.0
        offset_2 = phase_2.RescaleIntercept / 1000.0
    elif (
        hasattr(phase_1, "RealWorldValueMappingSequence")
        and hasattr(phase_1.RealWorldValueMappingSequence, "Item_1")
        and hasattr(phase_1.RealWorldValueMappingSequence.Item_1, "RealWorldValueSlope")
        and hasattr(
            phase_1.RealWorldValueMappingSequence.Item_1,
            "RealWorldValueIntercept",
        )
        and hasattr(phase_2, "RealWorldValueMappingSequence")
        and hasattr(phase_2.RealWorldValueMappingSequence, "Item_1")
        and hasattr(phase_2.RealWorldValueMappingSequence.Item_1, "RealWorldValueSlope")
        and hasattr(
            phase_2.RealWorldValueMappingSequence.Item_1,
            "RealWorldValueIntercept",
        )
    ):
        factor_1 = (
            phase_1.RealWorldValueMappingSequence.Item_1.RealWorldValueSlope / 1000.0
        )
        offset_1 = (
            phase_1.RealWorldValueMappingSequence.Item_1.RealWorldValueIntercept
            / 1000.0
        )
        factor_2 = (
            phase_2.RealWorldValueMappingSequence.Item_1.RealWorldValueSlope / 1000.0
        )
        offset_2 = (
            phase_2.RealWorldValueMappingSequence.Item_1.RealWorldValueIntercept
            / 1000.0
        )
    else:
        raise AttributeError(
            "SHIM: RescaleSlope or RescaleIntercept could not be determined. Shim analysis skipped"
        )

    data_type = ShimDataTypes.WRAPPED_PHASE_RADIANS

    # TODO:: in the original MATLAB code, the part above is a separate function from below, due to vendor specific
    # TODO:: acquisitions for B0 maps

    # relative roi size outer
    relative_roi_size_outer = 0.99

    # diameter of roi relative to phantom
    relative_roi_size = 0.95  # TODO make configurable

    # top part of roi to be ignored due to air bubble
    relative_ignore_top = 0.8  # TODO make configurable

    # find phantom ellipse
    # [x_axis_length, y_axis_length, center_x,center_y, phi]
    phantom_ellipse = retrieve_ellipse_parameters(image_data=magnitude.pixel_array)

    # create large mask
    # get top 5 vlaues avg
    # use taht to determine pos or neg

    # create masks

    mask_outer = elliptical_mask(
        radius_x=phantom_ellipse[3] * relative_roi_size,
        radius_y=phantom_ellipse[2] * relative_roi_size,
        center_x=phantom_ellipse[0],
        center_y=phantom_ellipse[1],
        angle=phantom_ellipse[4],
        dimension_x=phase_1.pixel_array.shape[0],
        dimension_y=phase_1.pixel_array.shape[1]
    ).astype(bool)

    mask = elliptical_mask(
        radius_x=phantom_ellipse[3] * relative_roi_size,
        radius_y=phantom_ellipse[2] * relative_roi_size,
        center_x=phantom_ellipse[0],
        center_y=phantom_ellipse[1],
        angle=phantom_ellipse[4],
        dimension_x=phase_1.pixel_array.shape[0],
        dimension_y=phase_1.pixel_array.shape[1],
        ignore_top=relative_ignore_top,
    ).astype(bool)

    def process_image(inner_image, inner_mask, inner_mask_outer, inner_factor, inner_offset, unwrap=False):
        inner_processed_image = np.copy(inner_image)
        inner_processed_image = inner_processed_image * inner_factor + inner_offset
        inner_processed_image = inner_processed_image * inner_mask_outer

        if unwrap:
            inner_processed_image = skimage.restoration.unwrap_phase(inner_processed_image)

        return inner_processed_image * mask

    # Calc avg value in outer ring and create new array with avg value as background and put cropped image in it

    if data_type is ShimDataTypes.WRAPPED_PHASE_RADIANS:
        processed_image_1 = process_image(phase_1.pixel_array, mask, mask_outer, factor_1, offset_1, unwrap=True)
        processed_image_2 = process_image(phase_2.pixel_array, mask, mask_outer, factor_2, offset_2, unwrap=True)
    else:
        processed_image_1 = process_image(phase_1.pixel_array, mask, mask_outer, factor_1, offset_1)
        processed_image_2 = process_image(phase_2.pixel_array, mask, mask_outer, factor_2, offset_2)

    phase_delta = (processed_image_2 - processed_image_1)
    phase_echo_delta = phase_2.EchoTime - phase_1.EchoTime

    magnet_tesla = phase_1.MagneticFieldStrength

    if data_type is ShimDataTypes.WRAPPED_PHASE_RADIANS:
        phase_delta_unwrapped = phase_delta * mask

        phase_delta_unwrapped_filename = str(
            Path(
                result.getDirName()
                + os.sep
                + "result_objects"
                + os.sep
                + "unwrapped_phase_map.png"
            ).absolute()
        )  # gets instantiated relative to working directory

        save_image_to_file(phase_delta_unwrapped, phase_delta_unwrapped_filename, "Unwrapped phase map cutout")


        gamma_rad_ms_T = (
            267513  # 42576 * 2 * pi;  gyromatic frequency in rad / ms * 1 / T
        )
        dB0_T = phase_delta_unwrapped / (gamma_rad_ms_T * phase_echo_delta)  # in Tesla
        dB0_ppm = dB0_T / (magnet_tesla * 1.0e6)  # ppm
    elif data_type is ShimDataTypes.B0_HZ:
        raise NotImplementedError("B0_HZ not yet implemented")
    elif data_type is ShimDataTypes.B0_PPM:
        raise NotImplementedError("B0_PPM not implemented yet")
    else:
        raise ValueError("Unknown data type in Shim analysis")

    # write B0 map to log file

    # calculate uniformity by iterating over masked image and comparing pixel values with largest/smallest
    smallest = 1.0e99  # smallest starts out large in order to find small values
    largest = -1.0e99  # vv
    # keep track of B0 values inside the mask; otherwise masked values will impact the mean
    B0_values = []
    for idx, row in enumerate(dB0_ppm):
        for idy, item in enumerate(row):
            if mask[idx][idy] == 1:
                B0_values.append(dB0_ppm[idx][idy])
                if dB0_ppm[idx][idy] < smallest:
                    smallest = item
                if dB0_ppm[idx][idy] > largest:
                    largest = item

    B0_uniformity_ppm = (largest - smallest) / np.mean(B0_values)
    print(f"B0 uniformity: {B0_uniformity_ppm}")
    result.addFloat("Shim", B0_uniformity_ppm)
    result.addObject("UnwrappedShimCutout", phase_delta_unwrapped_filename)


def save_image_to_file(image, name, title):
    plt.title(title)
    plt.imshow(image, cmap=plt.get_cmap("Greys_r"))

    plt.axis("off")
    plt.savefig(name, dpi=300)