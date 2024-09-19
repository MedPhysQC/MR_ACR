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
Analysis of image uniformity.

Workflow:
input: series with transversal "plain" slice of ACR phantom.
output: image uniformity as defined in the ACR guide.
       
1. Estimation of phantom centre and edge (see geometry_xy.py)
2. Define a large number of small circular ROIs covering the phantom.
3. Find the ROIs with smallest and largest average signal.
4. Output is the ghosting in percent as defined in the ACR guide:
   100 * (1 - (max_mean - min_mean) / (max_mean + min_mean))

Changelog:
    20240919: initial version 
"""

__version__ = '20230919'
__author__ = 'jkuijer'

from typing import List

import numpy as np

from util import (
    param_or_default,
    param_or_default_as_int,
    plot_rectangles_and_circles_on_image,
    get_points_in_circle,
    get_pixel_spacing,
    retrieve_ellipse_center,
    series_by_series_description,
    image_data_from_series,
    DicomSeriesList,
)


def image_uniformity(
    parsed_input: List[DicomSeriesList], result, action_config
) -> None:
    """
    percentual image uniformity = 100 * (1-((max_signal - min_signal) / (max_signal + min signal))
    """
    actionName = "image_uniformity"
    print( "> action " + actionName )

    series_description = action_config["params"]["image_uniformity_series_description"]
    print( "  search for configured SeriesDescription: " + series_description )

    number_of_img_in_series = param_or_default_as_int(
        action_config["params"],
        "image_uniformity_number_of_images_in_series",
        None
    )
    if number_of_img_in_series:
        print( "  search for configured number of images in series: " + str(number_of_img_in_series) )            

    series = series_by_series_description(series_description, parsed_input, number_of_img_in_series)
    image_number = int(param_or_default(action_config["params"], "image_uniformity_image_number", 6))

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

    [center_x, center_y] = retrieve_ellipse_center( image_data )
    print(
        "  detected phantom centre coordinates: "
        + "{:.0f}".format(center_x) + " " + "{:.0f}".format(center_y)
    )

    pixel_spacing = get_pixel_spacing(series)

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

    image_uniformity_result_filename = "image_uniformity_result.png"
    print("  write image_uniformity_result.png to: '" + image_uniformity_result_filename + "'")

    plot_rectangles_and_circles_on_image(
        image_data,
        [],
        [circle_1, circle_2, [(center_x, center_y), pixel_radius_phantom]],
        title="Image uniformity: search area, min and max",
        save_as=image_uniformity_result_filename,
    )
    result.addFloat("ImageUniformity_percent", image_uniformity_percent)
    result.addObject("ImageUniformity", image_uniformity_result_filename)
