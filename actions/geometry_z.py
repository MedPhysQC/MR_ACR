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
Analysis of gradient amplitude calibration of Z-gradient.

Workflow:
input: series with mid-sagital "plain" slice of ACR phantom. Ideally the phantom should have a small
       in-plane rotation of about 1-3 degrees to allow sub-sampling of the edge position.
output: length of phantom in Z-direction
       
1. Initial estimation of phantom centre
  a. Apply edge filter by kernel convolution
  b. Estimate phantom centre position by center-of-mass of it's edges.
2. Apply threshold at 20th-percentile to remove noise
3. Crop the image; use middle part only to avoid issues with curved phantom edges
   due to gradient non-lineairity.
4. Apply Radon transform
5. Find distance between the peaks in the Radon transformed image. This equals 
   distance between phantom edges.

Changelog:
    20240919: initial version
    20241213 / jkuijer: more selective peakfinding in sinogram to find low-intensity edge
    20250401 / jkuijer: more robust config param reading
"""

__version__ = '20250401'
__author__ = 'jkuijer'

from typing import List

import numpy as np
from scipy import ndimage

from util import (
    param_or_default_as_int,
    DicomSeriesList,
    plot_edges_on_image,
    interpolation_peak_offset,
    radon_transform,
    get_pixel_spacing,
    series_by_series_description,
    image_data_from_series,
)


def geometry_z(
    parsed_input: List[DicomSeriesList], result, action_config
) -> List[float]:
    # Z diameter
    actionName = "geometry_z"
    print( "> action " + actionName )

    series_description_z = action_config["params"]["geometry_z_series_description"]
    print( "  search for configured SeriesDescription: " + series_description_z )

    # Get dicom data
    series_z = series_by_series_description(
        series_description_z, parsed_input
    )

    print(
        "  matched with series number: "
        + str( series_z[0]["SeriesNumber"].value )
        + "\n  study_instance_uid: '"
        + series_z[0]["StudyInstanceUID"].value
        + "'\n  series_instance_uid: '"
        + series_z[0]["SeriesInstanceUID"].value
        + "'"
    )

    # get image data
    image_number_z = param_or_default_as_int(action_config["params"], "geometry_z_image_number", None)
    print( f"  image number = {image_number_z}" )

    image_data_z = image_data_from_series(series_z, image_number = image_number_z).astype("float")


    # Get pixel spacing
    pixel_spacing = get_pixel_spacing(series_z)

    # Filter image
    image_filtered = image_data_z.copy()

    # filter low signals out of the image to avoid false peak detection in sinogram
    noise_floor_threshold = np.percentile(image_data_z, 20)
    print("  apply noise floor threshold of percentile 20 : {:.0f}".format(noise_floor_threshold) )
    image_filtered[image_filtered < noise_floor_threshold] = 0

    
    # Calculate edge image
    # Define kernel for x differences
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    # Define kernel for y differences
    ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
    # Perform x convolution
    x=ndimage.convolve(image_filtered,kx)
    # Perform y convolution
    y=ndimage.convolve(image_filtered,ky)
    edges_z=np.hypot(x,y)

    # find approximate centre location of the phantom
    m = image_filtered != 0
    m = m / np.sum(np.sum(m))
    
    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)
    
    # expected values
    (szX,szY) = image_filtered.shape
    cx = np.sum(dx * np.arange(szX))
    cy = np.sum(dy * np.arange(szY))
    
    phantom_center_approximate_xy = np.array( [cx,cy] ).astype("int")
    
    # crop unwanted parts of the image
    # crop vertically, set everything outside center +- 35 mm to 0
    crop_size_mm = 35
    crop_size_px = round(crop_size_mm / pixel_spacing)
    
    edges_z_crop = edges_z.copy()
    # crop vertically, set everything outside center +- 35 mm to 0
    edges_z_crop[:, : phantom_center_approximate_xy[0] - crop_size_px] = 0
    edges_z_crop[:, phantom_center_approximate_xy[0] + crop_size_px :] = 0

    # crop horizontally, set everything INSIDE center +- 35 mm to 0. This will make the resulting sinogram less noisy
    edges_z_crop[
        phantom_center_approximate_xy[1]
        - crop_size_px : phantom_center_approximate_xy[1]
        + crop_size_px,
        :,
    ] = 0

    # generate sinogram to find the edges of the phantom
    max_deg_for_sinogram = 180.0
    sinogram = radon_transform(edges_z_crop, max_deg_for_sinogram)

    # find the coordinates for the two highest peaks
    # mask the sinogram so we only look at parallel, closest to perpendicular to columns
    search_angle_deg = 15
    start_px = int( szY / max_deg_for_sinogram * (90 - search_angle_deg) )
    end_px   = int( szY / max_deg_for_sinogram * (90 + search_angle_deg) )
    sinogram_masked = sinogram.copy()
    sinogram_masked[:, :start_px] = 0
    sinogram_masked[:, end_px:] = 0

    sino_max_per_row = np.sort(sinogram_masked)[
        :, szY-1
    ]  # this returns the final column (highest value per row) of the sorted rows of the sinogram

    # peak finder
    # find the column index for the first peak over 90% of the max value
    # this works well as we expect two high peaks in the data that are close to each other in value
    # a peak is where the following value is lower than the current
    threshold = 0.9 #0.60
    print("  threshold for peak finding in sinogram (relative to max): {:.2f}".format(threshold) )

    peak_first = [
        (index, value)
        for index, value in enumerate(sino_max_per_row[:-1])
        if sino_max_per_row[index + 1] < value
        if value > np.max(sino_max_per_row[:phantom_center_approximate_xy[1]]) * threshold
    ][0]

    # find the column index for the last peak over 90% of the max value
    # same as previous, but in reverse order
    peak_last = [
        (len(sino_max_per_row) - index - 1, value)
        for index, value in enumerate(np.flip(sino_max_per_row)[:-1])
        if np.flip(sino_max_per_row)[index + 1] < value
        if value > np.max(sino_max_per_row[phantom_center_approximate_xy[1]:]) * threshold
    ][0]

    # take the peak values to match with the sinogram
    peak_1_value = peak_first[1]
    peak_2_value = peak_last[1]

    # this yields two arrays where the first is the Y coordinate and the second is the X coordinate
    peak_1_coords = np.where(sinogram_masked == peak_1_value)
    peak_2_coords = np.where(sinogram_masked == peak_2_value)

    # rearrange into a format that is actually usable, interpolate the peaks
    peak_1_coords = np.array(
        [
            peak_1_coords[1][0],
            peak_1_coords[0][0]
            + interpolation_peak_offset(sino_max_per_row, peak_1_coords[0][0]),
        ]
    )
    peak_2_coords = np.array(
        [
            peak_2_coords[1][0],
            peak_2_coords[0][0]
            + interpolation_peak_offset(sino_max_per_row, peak_2_coords[0][0]),
        ]
    )

    # length of phantom in mm equals distance between peaks, converted from pixels to mm
    z_length_mm = ( peak_2_coords[1] - peak_1_coords[1] ) * pixel_spacing

    # both lines have slightly different angles of rotation due to curved edges on scan
    rotation_1 = peak_1_coords[0] * (180 / sinogram_masked.shape[0]) - 90
    rotation_2 = peak_2_coords[0] * (180 / sinogram_masked.shape[0]) - 90
    rotation = np.average([rotation_1, rotation_2])

    print("  phantom length: {:.2f}".format(z_length_mm) + " mm. Rotation: {:.2f}".format(rotation) + " deg.")

    # for plotting we need the tangent of the (rotation converted to radians)
    slope = np.tan(np.radians(rotation))

    # for plotting purposes calculate position of detected edge in image space
    # distance of peak to center of sinogram equals distance of phantom edge to center of image
    sinogram_center = np.array([sinogram.shape[0] // 2, sinogram.shape[1] // 2])

    # note that centre distance is sign sensitive
    peak_1_center_distance_px = peak_1_coords[1] - sinogram_center[1]
    peak_2_center_distance_px = peak_2_coords[1] - sinogram_center[1]

    # x-y coordinate pair of detected edges in image space
    edge_1 = (
        float(sinogram_center[1] - peak_1_center_distance_px * np.sin(np.radians(rotation))),
        float(sinogram_center[1] - peak_1_center_distance_px * np.cos(np.radians(rotation)))
    )
    edge_2 = (
        float(sinogram_center[1] - peak_2_center_distance_px * np.sin(np.radians(rotation))),
        float(sinogram_center[1] - peak_2_center_distance_px * np.cos(np.radians(rotation)))
    )


    # save result objects to directory relative to working directory
    z_result_image_filename = "geometry_z_result.png"
    z_edges_image_filename = "geometry_z_edges.png"

    print("  Write geometry_z_result.png to: '" + z_result_image_filename + "'")
    print("  Write geometry_z_edges.png to: '" + z_edges_image_filename + "'")


    # use negative slope because of matplotlib stuffs
    # coordinates obtained for the peaks must be subtracted from the max index (i.e. shape) for plotting
    plot_edges_on_image(
        #mask_to_coordinates(edges_z),
        None,
        image_data_z,
        title="GeometryZ fitted edge. Series " + str( series_z[0]["SeriesNumber"].value ),
        axlines=[
            {
                "xy1": edge_1,
                "slope": -slope,
                "color": "r",
                "linewidth": 0.75,
            },
            {
                "xy1": edge_2,
                "slope": -slope,
                "color": "r",
                "linewidth": 0.75,
            }
        ],
        axvlines=[
            {
                "x": float(phantom_center_approximate_xy[0] - crop_size_px),
                "color": "b",
                "linewidth": 0.75,
            },
            {
                "x": float(phantom_center_approximate_xy[0] + crop_size_px),
                "color": "b",
                "linewidth": 0.75,
            },
        ],
        markers_x_y=(
            np.array([
                float(phantom_center_approximate_xy[0]),
            ]),
            np.array([
                float(phantom_center_approximate_xy[1]),
            ]),
        ),
        save_as=z_result_image_filename,
    )

    plot_edges_on_image(
        None,
        edges_z,
        title="GeometryZ edges. Series " + str( series_z[0]["SeriesNumber"].value ),
        axlines=[
            {
                "xy1": edge_1,
                "slope": -slope,
                "color": "r",
                "linewidth": 0.75,
            },
            {
                "xy1": edge_2,
                "slope": -slope,
                "color": "r",
                "linewidth": 0.75,
            },
        ],
        axvlines=[
            {
                "x": float(phantom_center_approximate_xy[0] - crop_size_px),
                "color": "b",
                "linewidth": 0.75,
            },
            {
                "x": float(phantom_center_approximate_xy[0] + crop_size_px),
                "color": "b",
                "linewidth": 0.75,
            },
        ],
        lines_x_y=(
            np.array([ edge_1[0], edge_2[0] ]),
            np.array([ edge_1[1], edge_2[1] ]),
        ),
        save_as=z_edges_image_filename,
    )

    result.addFloat("GeometryZ_length_mm", z_length_mm)
    result.addFloat("GeometryZ_rotation_deg", rotation)
    result.addObject("GeometryZ_fit", z_result_image_filename)
    result.addObject("GeometryZ_edges", z_edges_image_filename)
    result.addString("GeometryZ_info", "GeometryZ on series " + str( series_z[0]["SeriesNumber"].value ) )
