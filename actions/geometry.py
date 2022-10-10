from pathlib import Path
from typing import List

import numpy as np
import os

from util import (
    DicomSeriesList,
    plot_edges_on_image,
    mask_to_coordinates,
    interpolation_peak_offset,
    radon_transform,
    detect_edges,
    get_pixel_spacing,
    acquisition_by_series_description,
    image_data_by_series_description,
    plot_ellipse_on_image,
    retrieve_ellipse_parameters,
)


def geometry_x_y(
    parsed_input: List[DicomSeriesList], action_config, result
) -> List[float]:
    series_description = action_config["params"]["geometry_x_y_series_description"]
    image_number = int(action_config["params"][
        "geometry_x_y_image_number"
    ])  # TODO check image number 0/1 indexed
    image_data = image_data_by_series_description(
        series_description, parsed_input, image_number=image_number, data_type="float"
    )  # edge detection wants floats

    acquisition = acquisition_by_series_description(series_description, parsed_input)

    actionName = "geometry_x_y"
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

    [
        x_center_px,
        y_center_px,
        x_axis_length_px,
        y_axis_length_px,
        phi,
    ] = retrieve_ellipse_parameters(image_data, mask_air_bubble=True)
    # retrieve pixel spacing [x,y], calculate diameter
    pixel_spacing = get_pixel_spacing(acquisition)
    x_diameter_mm = x_axis_length_px * 2 * pixel_spacing
    y_diameter_mm = y_axis_length_px * 2 * pixel_spacing

    # Save a plot
    geometry_xy_filename = str(
        Path(
            result.getDirName() + os.sep + "result_objects" + os.sep + "geometry_xy.png"
        ).absolute()
    )  # gets instantiated relative to working directory
    geometry_xy_filename_path = os.path.dirname(geometry_xy_filename)

    print("Write geometry_xy.png to: '" + geometry_xy_filename + "'")

    if not os.path.exists(geometry_xy_filename_path):
        os.makedirs(geometry_xy_filename_path)

    plot_ellipse_on_image(
        ellipse=[x_axis_length_px, y_axis_length_px, x_center_px, y_center_px, phi],
        acquisition=image_data,
        title="Geometry XY fitted ellipse",
        draw_axes=True,
        save_as=geometry_xy_filename,
    )  # TODO cirkel weglaten waar deze is weggelaten/gemaskeerd voor het fitten 2e ronde

    return [
        x_diameter_mm,
        y_diameter_mm,
        x_center_px,
        y_center_px,
        geometry_xy_filename,
    ]


def write_array_to_file(file_name, data_array):
    from array import array

    output_file = open(file_name, "wb")
    float_array = array("f", data_array)
    float_array.tofile(output_file)
    output_file.close()


def geometry_z(
    parsed_input: List[DicomSeriesList], action_config, result
) -> List[float]:
    # Z diameter
    series_description_z = action_config["params"]["geometry_z_series_description"]

    # Get image data
    image_data_z = image_data_by_series_description(
        series_description_z, parsed_input, data_type="float"
    )  # edge detection wants floats

    # Get dicom data
    acquisition_z = acquisition_by_series_description(
        series_description_z, parsed_input
    )

    actionName = "geometry_z"
    print(
        "action "
        + actionName
        + " called for "
        + "study_instance_uid: '"
        + acquisition_z[0]["StudyInstanceUID"].value
        + "' series_instance_uid: '"
        + acquisition_z[0]["SeriesInstanceUID"].value
        + "'"
    )

    # Get pixel spacing
    pixel_spacing = get_pixel_spacing(acquisition_z)

    # Filter image
    image_filtered = image_data_z.copy()

    # filter low signals out of the image
    image_filtered[image_filtered < image_filtered.max() / 4] = 0

    # Canny edge detection
    edges_z = detect_edges(
        image_filtered, sigma=1.5, low_threshold=3, high_threshold=100
    )

    # The most top right pixel with a value is the approximate top right of the phantom
    # Sum columns, resulting a row with sums. Select all greater than zero, and take last
    phantom_top_right_column = np.where(np.sum(edges_z, axis=0) > 0)[0][-1]

    # Edge detection can sometimes result in pixels outside the filtered image
    # Subtract 2 from found value
    phantom_top_right_column -= 2

    # Select top row
    selection = image_filtered[:, phantom_top_right_column] > 0
    phantom_top_right_row = np.where(selection)[0][0]

    # Create tuple
    phantom_top_right_xy = np.array([phantom_top_right_column, phantom_top_right_row])

    # the most bottom right pixel with a value is the approximate bottom right of the phantom
    phantom_bottom_right_row = np.where(np.sum(edges_z, axis=1) > 0)[0][-1]

    # Also account for offset
    phantom_bottom_right_row -= 2

    # Select bottom row
    selection = image_filtered[phantom_bottom_right_row] > 0
    phantom_bottom_right_column = np.where(selection)[0][-1]

    # Create tuple
    phantom_bottom_right_xy = np.array(
        [phantom_bottom_right_column, phantom_bottom_right_row]
    )

    # the specified width of the phantom is 190mm, meaning the square root of 190**2 + side length**2 is the length of the diagonal
    phantom_right_side_length_px = np.linalg.norm(
        phantom_top_right_xy - phantom_bottom_right_xy
    )
    phantom_top_side_specified_width_px = 190 / pixel_spacing
    # Calculate (approximate) slope
    phantom_approximate_slope = (
        phantom_top_right_xy[0] - phantom_bottom_right_xy[0]
    ) / (phantom_bottom_right_xy[1] - phantom_top_right_xy[1])
    # if the phantom was perpendicular resp. parallel to the x/y axes we can simply subtract the pixels for length and width
    # but because we have a (approximate) slope it is possible to correct
    x_delta = phantom_top_side_specified_width_px / 2
    y_delta = phantom_right_side_length_px / 2
    phantom_center_approximate_xy = np.array(
        [
            phantom_bottom_right_xy[0]
            - x_delta
            + (x_delta * phantom_approximate_slope),
            phantom_bottom_right_xy[1]
            - y_delta
            - (x_delta * phantom_approximate_slope),
        ]
    ).astype("int")
    # crop unwanted parts of the image
    pixel_range_for_cropping = round(45 / pixel_spacing)
    edges_z_crop = edges_z.copy()
    # crop vertically, set everything out side center +- 45 mm to 0
    edges_z_crop[:, : phantom_center_approximate_xy[0] - pixel_range_for_cropping] = 0
    edges_z_crop[:, phantom_center_approximate_xy[0] + pixel_range_for_cropping :] = 0

    # crop horizontally, set everything INSIDE center +- 45 mm to 0. This will make the resulting sinogram less noisy
    edges_z_crop[
        phantom_center_approximate_xy[1]
        - pixel_range_for_cropping : phantom_center_approximate_xy[1]
        + pixel_range_for_cropping,
        :,
    ] = 0

    # generate sinogram to find the edges of the phantom
    max_deg_for_sinogram = 180.0
    sinogram = radon_transform(edges_z_crop, max_deg_for_sinogram)

    # find the coordinates for the two highest peaks
    # mask the sinogram so we only look at parallel, closest to perpendicular to columns
    sinogram_masked = sinogram.copy()
    sinogram_masked[:, :212] = 0
    sinogram_masked[:, 301:] = 0

    sino_max_per_row = np.sort(sinogram_masked)[
        :, 511
    ]  # this returns the final column (highest value per row) of the sorted rows of the sinogram

    # peak finder
    # find the column index for the first peak over 90% of the max value
    # this works well as we expect two high peaks in the data that are close to each other in value
    # a peak is where the following value is lower than the current
    threshold = 0.60
    peak_first = [
        (index, value)
        for index, value in enumerate(sino_max_per_row[:-1])
        if sino_max_per_row[index + 1] < value
        if value > np.max(sino_max_per_row) * threshold
    ][0]

    # find the column index for the last peak over 90% of the max value
    # same as previous, but in reverse order
    peak_last = [
        (len(sino_max_per_row) - index - 1, value)
        for index, value in enumerate(np.flip(sino_max_per_row)[:-1])
        if np.flip(sino_max_per_row)[index + 1] < value
        if value > np.max(sino_max_per_row) * threshold
    ][0]

    # take the peak values to match with the sinogram
    peak_1_value = peak_first[1]
    peak_2_value = peak_last[1]

    # this  yields two arrays where the first is the Y coordinate and the second is the X coordinate
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

    # distance to center of sinogram is distance of line to center image
    sinogram_center = np.array([sinogram.shape[0] // 2, sinogram.shape[1] // 2])

    angle = np.cos(np.radians(max_deg_for_sinogram / sinogram.shape[0] * (sinogram_center[0] - peak_1_coords[0])))
    peak_1_center_distance_pixels = np.absolute((sinogram_center[1] - peak_1_coords[1]) / angle)

    angle = np.cos(np.radians(max_deg_for_sinogram / sinogram.shape[0] * (sinogram_center[0] - peak_2_coords[0])))
    peak_2_center_distance_pixels = np.absolute((sinogram_center[1] - peak_2_coords[1]) / angle)


    # length of phantom in mm is both lengths added and corrected for pixel spacing
    z_length_mm = (
        peak_1_center_distance_pixels + peak_2_center_distance_pixels
    ) * pixel_spacing

    # both lines have slightly different angles of rotation due to curved edges on scan
    rotation_1 = peak_1_coords[0] * (180 / sinogram_masked.shape[0]) - 90
    rotation_2 = peak_2_coords[0] * (180 / sinogram_masked.shape[0]) - 90
    rotation = np.average([rotation_1, rotation_2])

    # for plotting we need the tangent of the (rotation converted to radians)
    slope = np.tan(np.radians(rotation))
    # save result objects to directory relative to working directory
    z_result_image_filename = str(
        Path(
            result.getDirName()
            + os.sep
            + "result_objects"
            + os.sep
            + "geometry_z_result.png"
        ).absolute()
    )  # gets instantiated relative to working directory
    z_edges_image_filename = str(
        Path(
            result.getDirName()
            + os.sep
            + "result_objects"
            + os.sep
            + "geometry_z_edges.png"
        ).absolute()
    )  # gets instantiated relative to working directory

    print("Write geometry_z_result.png to: '" + z_result_image_filename + "'")
    print("Write geometry_z_edges.png to: '" + z_edges_image_filename + "'")

    z_result_image_filename_path = os.path.dirname(z_result_image_filename)
    if not os.path.exists(z_result_image_filename_path):
        os.makedirs(z_result_image_filename_path)

    # use negative slope because of matplotlib stuffs
    # coordinates obtained for the peaks must be subtracted from the max index (i.e. shape) for plotting
    # TODO document all extra lines plotted
    plot_edges_on_image(
        mask_to_coordinates(edges_z),
        image_data_z,
        title=f"Z Result, cropping highlighted, acq date: {acquisition_z[0].SeriesDate}",
        axlines=[
            {
                "xy1": (
                    sinogram_center[0],
                    float(sinogram.shape[0] - peak_1_coords[1]),
                ),
                "slope": -slope,
                "color": "r",
                "linewidth": 0.75,
            },
            {
                "xy1": (
                    sinogram_center[0],
                    float(sinogram.shape[0] - peak_2_coords[1]),
                ),
                "slope": -slope,
                "color": "r",
                "linewidth": 0.75,
            },
        ],
        axvlines=[
            {
                "x": float(phantom_center_approximate_xy[0] - pixel_range_for_cropping),
                "color": "b",
                "linewidth": 0.75,
            },
            {
                "x": float(phantom_center_approximate_xy[0] + pixel_range_for_cropping),
                "color": "b",
                "linewidth": 0.75,
            },
            {
                "x": float(phantom_center_approximate_xy[0]),
                "color": "y",
                "linewidth": 0.75,
            },
        ],
        axhlines=[
            {
                "y": float(phantom_center_approximate_xy[1] - pixel_range_for_cropping),
                "color": "b",
                "linewidth": 0.75,
            },
            {
                "y": float(phantom_center_approximate_xy[1] + pixel_range_for_cropping),
                "color": "b",
                "linewidth": 0.75,
            },
            {
                "y": float(phantom_center_approximate_xy[1]),
                "color": "g",
                "linewidth": 0.75,
            },
        ],
        save_as=z_result_image_filename,
    )

    plot_edges_on_image(
        None,
        edges_z,
        title=f"Z Result, detected edges, cropping highlighted, acq date: {acquisition_z[0].SeriesDate}",
        axlines=[
            {
                "xy1": (
                    float(sinogram_center[0]),
                    float(sinogram.shape[0] - peak_1_coords[1]),
                ),
                "slope": -slope,
                "color": "r",
                "linewidth": 0.75,
            },
            {
                "xy1": (
                    float(sinogram_center[0]),
                    float(sinogram.shape[0] - peak_2_coords[1]),
                ),
                "slope": -slope,
                "color": "r",
                "linewidth": 0.75,
            },
        ],
        axvlines=[
            {
                "x": float(phantom_center_approximate_xy[0] - pixel_range_for_cropping),
                "color": "b",
                "linewidth": 0.75,
            },
            {
                "x": float(phantom_center_approximate_xy[0] + pixel_range_for_cropping),
                "color": "b",
                "linewidth": 0.75,
            },
        ],
        axhlines=[
            {
                "y": float(phantom_center_approximate_xy[1] - pixel_range_for_cropping),
                "color": "b",
                "linewidth": 0.75,
            },
            {
                "y": float(phantom_center_approximate_xy[1] + pixel_range_for_cropping),
                "color": "b",
                "linewidth": 0.75,
            },
        ],
        save_as=z_edges_image_filename,
    )

    return [z_length_mm, rotation, z_result_image_filename, z_edges_image_filename]


def geometry(parsed_input: List[DicomSeriesList], result, action_config) -> None:
    [x_diameter_mm, y_diameter_mm, x_center, y_center, figure_filename] = geometry_x_y(
        parsed_input, action_config, result
    )
    [
        z_length_mm,
        rotation,
        z_result_image_filename,
        z_edges_image_filename,
    ] = geometry_z(parsed_input, action_config, result)

    result.addString(
        "GeometryXYCenter",
        f"Centre location at x:"
        + "{:.2f}".format(x_center)
        + ", y"
        + "{:.2f}".format(y_center)
        + "",
    )

    # TODO addstring for series /slice of location for xy and z analysis
    result.addFloat("GeometryX", x_diameter_mm)
    result.addFloat("GeometryY", y_diameter_mm)
    result.addFloat("GeometryZ", z_length_mm)
    result.addFloat("GeometryInPlaneRotation", rotation)

    result.addObject("GeometryXY", figure_filename)
    result.addObject("GeometryZResults", z_result_image_filename)
    result.addObject("GeometryZEdgesOnly", z_edges_image_filename)
