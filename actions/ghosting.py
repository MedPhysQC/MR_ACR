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
Analysis of image ghosting.

Workflow:
input: series with transversal "plain" slice of ACR phantom. The image should
       *NOT* be corrected for geometrical distortion by gradient non-linearity,
       as this would interpolate (or resample) the signal between pixels, thereby
       affecting the noise amplitude. Parallel imaging techniques such as
       SENSE, GRAPPA or ASSET should *NOT* be used because these affect the 
       estimation of noise in a background ROI.
output: ghosting as defined in the ACR guide.
       
1. Estimation of phantom centre and edge (see geometry_xy.py)
2. Define ROIs for
  a. signal: centre of phantom
  b. noise: background signal in corners of image
  c. ghosting: background signal alongside the phantom
3. Output is the ghosting in percent as defined in the ACR guide:
   ( mean(ROI_ghost) - mean(ROI_noise) ) / 2*mean(ROI_phantom) * 100%

Changelog:
    20240919: initial version 
"""

__version__ = '20240919'
__author__ = 'jkuijer'

from typing import List

import numpy as np

from util import (
    DicomSeriesList,
    param_or_default,
    param_or_default_as_int,
    series_by_series_description,
    image_data_from_series,
    get_pixel_spacing,
    retrieve_ellipse_center,
    get_ghosting_rois_pixel_values,
    get_background_rois_pixel_values,
    get_pixel_values_circle,
    plot_rectangles_and_circles_on_image,
)


def ghosting(parsed_input: List[DicomSeriesList], result, action_config) -> None:
    """
    signal ghosting percentage = 100 * (abs(ghost_signal - noise_signal) / (2* mean_signal))
    """
    actionName = "ghosting"
    print( "> action " + actionName )
    
    series_description = action_config["params"]["ghosting_series_description"]
    ghosting_roi_short_side_mm = float(action_config["params"]["ghosting_roi_short_side_mm"])
    ghosting_roi_long_side_mm = float(action_config["params"]["ghosting_roi_long_side_mm"])
    noise_roi_sides_mm = float(action_config["params"]["ghosting_noise_roi_sides_mm"])
    phantom_roi_mm = float(action_config["params"]["ghosting_phantom_roi_diameter_mm"])
    roi_shift_mm = float(action_config["params"]["ghosting_background_roi_shift_mm"])

    print( "  search for configured SeriesDescription: " + series_description )

    number_of_img_in_series = param_or_default_as_int(
        action_config["params"],
        "ghosting_number_of_images_in_series",
        None
    )
    if number_of_img_in_series:
        print( "  search for configured number of images in series: " + str(number_of_img_in_series) )            


    series = series_by_series_description(series_description, parsed_input, number_of_img_in_series)
    image_number = int(param_or_default(action_config["params"], "ghosting_image_number", 6))

    print(
        "  matched with series number: "
        + str( series[0]["SeriesNumber"].value )
        + "\n  study_instance_uid: '"
        + series[0]["StudyInstanceUID"].value
        + "'\n  series_instance_uid: '"
        + series[0]["SeriesInstanceUID"].value
        + "'"
    )

    if image_number:
        print( "  image number = {:0.0f}".format(image_number))
    image_data = image_data_from_series( series, image_number=image_number )

    # retrieve pixel spacing to get the pixel values from the correct positions, verify square pixels
    pixel_spacing = get_pixel_spacing(series)

    [center_x, center_y] = retrieve_ellipse_center( image_data )

    print(
        "  detected phantom centre coordinates: "
        + "{:.0f}".format(center_x) + " " + "{:.0f}".format(center_y)
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

    ghosting_rows_percent = (
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
    ghosting_columns_percent = (
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
    
    image_ghosting_result_filename = "image_ghosting_result.png"
    print("  write ghosting ROI image to: '" + image_ghosting_result_filename + "'")

    plot_rectangles_and_circles_on_image(
        np.log10(image_data+1),
        [*ghosting_roi_definitions_for_plotting, *noise_roi_definitions_for_plotting],
        [[(center_x, center_y), phantom_roi_mm / 2]],
        title="Ghosting ROIs on log-transformed image",
        save_as=image_ghosting_result_filename,
    )
    result.addFloat("GhostingRow_percent", ghosting_rows_percent)
    result.addFloat("GhostingCol_percent", ghosting_columns_percent)
    
    if "InPlanePhaseEncodingDirection" in series[0]:
        PE_dir = str( series[0]["InPlanePhaseEncodingDirection"].value )
        result.addString("GhostingPhaseEncodingDirection", PE_dir)

    result.addObject("Ghosting", image_ghosting_result_filename)