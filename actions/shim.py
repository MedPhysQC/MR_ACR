from enum import Enum
from typing import List

import numpy as np

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

    # magnitude index = 0; used for determining center of the phantom
    magnitude = acquisition[0]

    # phase1 index = 2, phase2 index=3
    phase_1 = acquisition[2]
    phase_2 = acquisition[3]

    # check if acquisition has FFE in 2001,1020
    private_2001_1020 = magnitude[0x20011020].value.strip()
    if private_2001_1020 != "FFE":
        raise ValueError(
            f"SHIM: Could not detect Philips double echo FFE. Expected 'FFE'; got {private_2001_1020}"
        )

    # check if there are 4 images
    if len(acquisition) != 4:
        raise ValueError(f"SHIM: Expected 4 images, found {len(acquisition)}")

    # convert pixel values to radians

    if hasattr(phase_1, "RescaleSlope") and hasattr(phase_2, "RescaleSlope"):
        factor_1 = phase_1.RescaleSlope / 1000.0
        offset_1 = 0  # TODO verify why this is 0
        factor_2 = phase_2.RescaleSlope / 1000.0
        offset_2 = 0
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

    phase_delta = ((phase_2.pixel_array * factor_2) - offset_2) - (
        (phase_1.pixel_array * factor_1) - offset_1
    )
    phase_echo_delta = phase_2.EchoTime - phase_1.EchoTime

    data_type = ShimDataTypes.WRAPPED_PHASE_RADIANS

    # TODO:: in the original MATLAB code, the part above is a separate function from below, due to vendor specific
    # TODO:: acquisitions for B0 maps

    # diameter of roi relative to phantom
    relative_roi_size = 0.95  # TODO make configurable

    # top part of roi to be ignored due to air bubble
    relative_ignore_top = 0.8  # TODO make configurable

    # find phantom ellipse
    # [x_axis_length, y_axis_length, center_x,center_y, phi]
    phantom_ellipse = retrieve_ellipse_parameters(image_data=magnitude.pixel_array)

    # create mask
    mask = elliptical_mask(
        radius_x=phantom_ellipse[3] * relative_roi_size,
        radius_y=phantom_ellipse[2] * relative_roi_size,
        center_x=phantom_ellipse[0],
        center_y=phantom_ellipse[1],
        angle=phantom_ellipse[4],
        dimension_x=phase_1.pixel_array.shape[0],
        dimension_y=phase_1.pixel_array.shape[1],
        ignore_top=relative_ignore_top,
    )

    # apply the mask to the phase delta
    masked_phase_delta = mask * phase_delta

    magnet_tesla = phase_1.MagneticFieldStrength

    if data_type is ShimDataTypes.WRAPPED_PHASE_RADIANS:
        phase_delta_unwrapped = np.unwrap(masked_phase_delta) * mask
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
