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
Analysis of gradient amplitude calibration of X- and Y-gradients.

Workflow:
input: series with transversal "plain" slice of ACR phantom
output: diameter of phantom along X- and Y-directions
       
1. Initial estimation of phantom centre and edge
  a. Apply Canny edge detection. Threshold with hysteresis at 80 and 90 percentiles of image intensity.
  b. Fit ellipse to phantom edge
2. Remove edges at the air bubble on the top of the phantom and repeat step 1.
3. Remove outliers (edges above some distance from the fitted ellipse) and repeat step 1.
4. Return the diameter in X and Y directions of the fitted ellipse.

Changelog:
    20240919: initial version 
"""

__version__ = '20240919'
__author__ = 'jkuijer'

from typing import List
import numpy as np
from scipy import fft


from util import (
    DicomSeriesList,
    param_or_default,
    plot_edges_on_image,
    get_pixel_spacing,
    series_by_series_description,
    image_data_by_series_description,
    plot_ellipse_on_image,
    retrieve_ellipse_parameters,
)


def geometry_xy(
    parsed_input: List[DicomSeriesList], result, action_config
) -> List[float]:

    actionName = "geometry_xy"
    print( "> action " + actionName )

    series_description = action_config["params"]["geometry_xy_series_description"]
    print( "  search for configured SeriesDescription: " + series_description )

    image_number = int(
        param_or_default(action_config["params"], "geometry_xy_image_number", 6)
    )
    
    mask_air_bubble_mm = float( 
        param_or_default(action_config["params"], "geometry_xy_air_bubble_mask_size_mm", 50)
    )
    
    image_data = image_data_by_series_description(
        series_description, parsed_input, image_number=image_number, data_type="float"
    )  # edge detection wants floats

    series = series_by_series_description(series_description, parsed_input)

    print(
        "  matched with series number: "
        + str( series[0]["SeriesNumber"].value )
        + "\n  study_instance_uid: '"
        + series[0]["StudyInstanceUID"].value
        + "'\n  series_instance_uid: '"
        + series[0]["SeriesInstanceUID"].value
        + "'"
    )
    
    upsampling_factor = 2
    if upsampling_factor > 1:
        # sinc interpolation
        x = fft.fft2(image_data)
        sz = len(x) # matrix size
        sz2 = sz // 2
        # empty matrix for zero-padded k-space
        x_pad = np.zeros( (sz*upsampling_factor, sz*upsampling_factor), dtype=complex )
        # spatial frequency zero is in corner of matrix, thus copy corners to new matrix
        x_pad[0:sz2,0:sz2] = x[0:sz2,0:sz2]
        x_pad[-sz2:,0:sz2] = x[-sz2:,0:sz2]
        x_pad[-sz2:,-sz2:] = x[-sz2:,-sz2:]
        x_pad[0:sz2,-sz2:] = x[0:sz2,-sz2:]
        y = fft.ifft2((x_pad))
        image_data = abs(y)
    
    # retrieve pixel spacing [x,y]
    pixel_spacing = get_pixel_spacing(series) / upsampling_factor
    mask_air_bubble_px = mask_air_bubble_mm / pixel_spacing
    print("  air bubble mask size: {:.1f}".format(mask_air_bubble_mm) + " mm")
    
    
    [
        ( x_center_px, y_center_px, x_axis_length_px, y_axis_length_px, phi ),
        edges,
        edge_coord_masked
    ] = retrieve_ellipse_parameters(
        image_data,
        mask_air_bubble_px=mask_air_bubble_px,
        return_edge_data=True
    )

    # calculate diameter in mm
    x_diameter_mm = x_axis_length_px * 2 * pixel_spacing
    y_diameter_mm = y_axis_length_px * 2 * pixel_spacing

    print(
        "  detected phantom centre coordinates: "
        + "{:.0f}".format(x_center_px) + " " + "{:.0f}".format(y_center_px)
        + "\n  fitted ellipse: "
        + "diam X = {:.4f}".format(x_diameter_mm) + " Y = {:.4f}".format(y_diameter_mm) + " mm, "
        + "angle = {:.4f}".format(np.degrees(phi)) + " deg"
    )

    # Save image with detected edge points
    geometry_xy_contour_filename = "geometry_xy_edgepoints.png"
    print("  Write contour image to: '" + geometry_xy_contour_filename + "'")

    # create the plot
    plot_edges_on_image(
        edge_coord_masked,
        image_data,
        title="Geometry XY detected edges. Series:"
          + str( series[0]["SeriesNumber"].value )
          + "#{:02.0f}".format( image_number ),
        save_as=geometry_xy_contour_filename
    )

    # Save a plot
    geometry_xy_filename = "geometry_xy.png"
    print("  Write fitted ellipse to: '" + geometry_xy_filename + "'")


    plot_ellipse_on_image(
        [x_center_px, y_center_px, x_axis_length_px, y_axis_length_px, phi],
        image_data,
        title="Geometry XY fitted ellipse. Series:"
          + str( series[0]["SeriesNumber"].value )
          + "#{:02.0f}".format( image_number ),
        draw_axes=True,
        save_as=geometry_xy_filename,
    )

    # write results
    result.addFloat("GeometryXY_diameter_X_mm", x_diameter_mm)
    result.addFloat("GeometryXY_diameter_Y_mm", y_diameter_mm)
    result.addObject("GeometryXY_fit", geometry_xy_filename)
    result.addObject("GeometryXY_edges", geometry_xy_contour_filename)

    result.addString(
        "GeometryXY_series_info",
        "Analysis on series "
        + str( series[0]["SeriesNumber"].value )
        + "#{:02.0f}".format( image_number )
    )

    result.addString(
        "GeometryXY Centre",
        "Centre location at "
        + "{:.2f}".format(x_center_px)
        + " "
        + "{:.2f}".format(y_center_px),
    )
