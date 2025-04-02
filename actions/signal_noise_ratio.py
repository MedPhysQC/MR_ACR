# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Analysis of image SNR.

Workflow:
input: series with transversal "plain" slice of ACR phantom. For phased array 
       coils: use the combined image. The image should
       *NOT* be corrected for geometrical distortion by gradient non-linearity,
       as this would interpolate (or resample) the signal between pixels, thereby
       affecting the noise amplitude. Parallel imaging techniques such as
       SENSE, GRAPPA or ASSET should *NOT* be used because these affect the 
       estimation of noise in a background ROI.
output: SNR, with noise estimated from background ROIs.
       
1. Estimation of phantom centre and edge (see geometry_xy.py)
2. Define ROIs for
  a. signal: centre of phantom
  b. noise: background signal in corners of image
3. Output is the ghosting in percent as defined in the ACR guide:
   0.655 * (signal_phantom / StdDev_background)

Changelog:
    20240919: initial version 
    20250401 / jkuijer: more robust config param reading
"""

__version__ = '20250401'
__author__ = 'jkuijer'

from typing import List

import numpy as np

from util import (
    DicomSeriesList,
    param_or_default,
    param_or_default_as_int,
    param_or_default_as_float,
    series_by_series_description,
    image_data_from_series,
    retrieve_ellipse_center,
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

    Valid if the mean / standard deviation of the background roi is 1.91

    """
    default_signal_roi_diameter_mm = 150
    default_noise_roi_sides_mm = 7
    default_background_roi_shift_mm = 108
    default_image_number = 6

    actionName = "signal_noise_ratio"
    print( "> action " + actionName )

    signal_roi_diameter_mm = param_or_default_as_float(
        action_config["params"],
        "snr_signal_roi_diameter_mm",
        default_signal_roi_diameter_mm
    )
    noise_roi_sides_mm = param_or_default_as_float(
        action_config["params"],
        "snr_noise_roi_sides_mm",
        default_noise_roi_sides_mm
    )
    background_roi_shift_mm = param_or_default_as_float(
        action_config["params"],
        "snr_background_roi_shift_mm",
        default_background_roi_shift_mm
    )

    series_description = action_config["params"]["snr_series_description"]
    print( "  number of series:  " + str(len(parsed_input)) )
    print( "  search for configured SeriesDescription: " + series_description )
    
    number_of_img_in_series = param_or_default_as_int(
        action_config["params"],
        "snr_number_of_images_in_series",
        None
    )
    if number_of_img_in_series:
        print( "  search for configured number of images in series: " + str(number_of_img_in_series) )            
    
    series = series_by_series_description(series_description, parsed_input, number_of_img_in_series)
    image_number = int(param_or_default(action_config["params"], "snr_image_number", default_image_number))

    print(
        "  matched for series number: "
        + str( series[0]["SeriesNumber"].value )
        + "\n  study_instance_uid: '"
        + series[0]["StudyInstanceUID"].value
        + "' \n  series_instance_uid: '"
        + series[0]["SeriesInstanceUID"].value
        + "'"
    )

    image_data = image_data_from_series( series, image_number )

    # determine center coordinates of phantom
    [center_x, center_y] = retrieve_ellipse_center(
        image_data
    )
    print(
        "  detected phantom centre coordinates: "
        + "{:.0f}".format(center_x) + " " + "{:.0f}".format(center_y)
    )

    # retrieve pixel spacing to get the pixel values from the correct positions
    pixel_spacing = get_pixel_spacing(series)

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
    print("  SNR = {:.0f}".format(signal_to_noise_ratio))

    # save result objects to directory relative to working directory
    snr_result_image_filename = "snr_rois_result.png"
    print("  Write SNR ROI image to: '" + snr_result_image_filename + "'")

    plot_rectangles_and_circles_on_image(
        image_data,
        rectangles=background_rois,
        circles=[[(center_x, center_y), signal_roi_diameter_mm / 2]],
        title="SNR",
        save_as=snr_result_image_filename,
    )

    result.addFloat("SNR_CombinedCoils", signal_to_noise_ratio)
    result.addObject("SNR_rois", snr_result_image_filename)
