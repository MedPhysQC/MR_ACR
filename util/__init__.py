import random
from enum import Enum
from typing import List, Tuple, Any, Union
import logging as log

import numpy

from wad_qc.modulelibs import wadwrapper_lib as wad_lib
import pydicom
import numpy as np
import matplotlib

matplotlib.use("Agg")  # noqa: backend selection must be done before plt import
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle

from skimage import feature
from skimage.transform import radon
from skimage.draw import ellipse
from numpy.linalg import eig

# general DPI setting for saving figures
DPI = 300

# TODO rename param acquisition to image_data
def plot_rectangles_and_circles_on_image(
    image_data, rectangles, circles, title=None, save_as=None
):
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    for rectangle in rectangles:
        rect_plot = Rectangle(
            xy=(
                (rectangle[0][0] - rectangle[1] / 2),
                (rectangle[0][1] - rectangle[2] / 2),
            ),
            width=rectangle[1],
            height=rectangle[2],
        )
        rect_plot.set_edgecolor("red")
        rect_plot.set_fill(False)
        ax.add_artist(rect_plot)
    for circle in circles:
        circle_plot = Circle(xy=circle[0], radius=circle[1], linewidth=0.75)
        circle_plot.set_edgecolor("orange")
        circle_plot.set_fill(False)
        ax.add_artist(circle_plot)
    if title:
        plt.title(title)
    plt.imshow(image_data, cmap=plt.get_cmap("Greys_r"))
    if save_as:
        plt.axis("off")
        plt.savefig(save_as, dpi=DPI)
    else:
        plt.show(block=True)
    plt.clf()
    plt.close()


def plot_ellipse_on_image(
    ellipse, acquisition, title=None, draw_axes=False, save_as: str = None
):
    """

    :param ellipse: ellipse parameters [width, height, x center, y center, rotation]
    :param acquisition:
    :param title:
    :param draw_axes: whether or not to draw the axes
    :param save_as: absolute file path including extension. if used, a plot is not shown to the user
    """
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ellipse_plot = Ellipse(
        xy=(ellipse[2], ellipse[3]),
        width=ellipse[0] * 2,
        height=ellipse[1] * 2,
        angle=ellipse[4],
    )
    ellipse_plot.set_edgecolor("red")
    ellipse_plot.set_linewidth(0.7)
    ellipse_plot.set_fill(False)
    ax.add_artist(ellipse_plot)
    if draw_axes:
        x_1, y_1 = [ellipse[2] - ellipse[0], ellipse[2] + ellipse[0]], [
            ellipse[3],
            ellipse[3],
        ]
        x_2, y_2 = [ellipse[2], ellipse[2]], [
            ellipse[3] - ellipse[1],
            ellipse[3] + ellipse[1],
        ]

        plt.plot(x_1, y_1, x_2, y_2, linewidth=0.5, color="r")
    if title:
        plt.title(title)
    plt.imshow(acquisition, cmap=plt.get_cmap("Greys_r"))

    if save_as:
        plt.axis("off")
        plt.savefig(save_as, dpi=DPI)
    else:
        plt.show(block=True)
    plt.clf()
    plt.close()


def plot_histogram(image, bins=10, title=None):
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    if title:
        plt.title(title)

    cnts, bins = np.histogram(np.hstack(image), bins=bins)
    ax.bar(bins[:-1] + np.diff(bins) / 2, cnts, np.diff(bins))
    # plt.bar(image)
    plt.show(block=True)
    plt.clf()
    plt.cla()


def plot_image(acquisition, title=None, colormap="Greys_r"):
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    if title:
        plt.title(title)
    plt.imshow(acquisition, cmap=plt.get_cmap(colormap))
    plt.show(block=True)


def plot_images(acquisitions, title=None, colormap="Greys_r"):
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    if title:
        plt.title(title)
    for acquisition in acquisitions:
        plt.imshow(acquisition, cmap=plt.get_cmap(colormap))
    plt.show(block=True)


def plot_edges_on_image(
    edges_x_y,
    acquisition,
    title=None,
    axlines=None,
    axvlines=None,
    axhlines=None,
    save_as=None,
):
    """

    :param axhlines: list of dicts containing axhlines params
    :param save_as:
    :param axvlines: list of dicts containing axvlines params
    :param edges_x_y: [[x coords],[y coords]]
    :param acquisition: pixel data as found on a pydicom object
    :param title: ...
    :param axlines:  list of dicts containing axlines params
    """
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    if edges_x_y:
        plt.scatter(edges_x_y[0], edges_x_y[1], s=1, c="r", alpha=0.3)
    if title:
        plt.title(title)
    plt.imshow(acquisition, cmap=plt.get_cmap("Greys_r"))

    if axlines:
        for line in axlines:
            plt.axline(**line)
    if axvlines:
        for line in axvlines:
            plt.axvline(**line)
    if axhlines:
        for line in axhlines:
            plt.axhline(**line)
    if save_as:
        plt.axis("off")
        plt.savefig(save_as, dpi=DPI)
    else:
        plt.show(block=True)

    plt.clf()
    plt.close()


def plot_ellipse_on_edges(ellipse, edges_x_y, title=None):
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ellipse_plot = Ellipse(
        xy=(ellipse[2], ellipse[3]),
        width=ellipse[0] * 2,
        height=ellipse[1] * 2,
        angle=ellipse[4],
    )
    ellipse_plot.set_edgecolor("red")
    ellipse_plot.set_fill(False)
    ax.add_artist(ellipse_plot)
    if title:
        plt.title(title)
    plt.scatter(edges_x_y[0], edges_x_y[1], s=1, c="y", alpha=0.3)

    plot_limit = max(max(edges_x_y[0]), max(edges_x_y[1]))
    plt.xlim([0, plot_limit + 10])
    plt.ylim([0, plot_limit + 10])
    #
    plt.show(block=True)


class DicomSeriesList(list):
    """
    Contains a list of pydicom DataSets, whose path is specified in filelist
    Sorted by InstanceNumber (0020,0013)
    Subclass of list to facilitate the SeriesDescription property
    """

    def __init__(self, filelist):
        super().__init__()
        for file in filelist:
            ds = pydicom.dcmread(file)
            self.append(ds)
        self.sort(key=lambda x: int(x["InstanceNumber"].value))

    @property
    def series_description(self):
        return self[0]["SeriesDescription"].value.strip()


def parse_inputs(series_filelist: List[List[str]]) -> List[DicomSeriesList]:
    """
    Creates the following datastructure:
    [ series_number_1:
        - series description
        - [pydicom DataSet of instance_number_1, pydicom DataSet of instance_number_2...] ],
    [series_number_2:
        - series description
        - [pydicom DataSet of instance_number_1, pydicom DataSet of instance_number_2...] ]

    It is a nested list, sorted on SeriesNumber (0020,0011)
    Does not make use of wad_qc.modulelibs.wadwrapper_lib.prepareInput() to skip the preprocessing that is done there
    """
    dcm_series = []
    for item in series_filelist:
        dcm_series.append(DicomSeriesList(item))
    dcm_series.sort(key=lambda x: int(x[0]["SeriesNumber"].value))
    return dcm_series


def image_data_by_series_description(
    series_description: str,
    parsed_input: list,
    image_number: int = None,
    data_type=None,
) -> np.ndarray:
    """
    Retrieve only the pixel data for a series, or if needed one volume of the series

    :param data_type: set the pixel values in the returned array to a specific type i.e. 'float', 'int'. See numpy docs
           for supported types
    :param image_number: specify an instance number.If None returns all slices, ignored for single slice
    :param series_description:
    :param parsed_input:
    :return: np.array with pixel data
    """
    for dcm_list in parsed_input:
        if dcm_list.series_description == series_description:
            image_data = (
                np.array(
                    [
                        dcm_dataset.pixel_array
                        for dcm_dataset in dcm_list
                        if hasattr(dcm_dataset, "pixel_array")
                    ]
                )
                if not data_type
                else np.array(
                    [
                        dcm_dataset.pixel_array
                        for dcm_dataset in dcm_list
                        if hasattr(dcm_dataset, "pixel_array")
                    ]
                ).astype(data_type)
            )
            if len(image_data) == 1:
                return image_data[0]
            return image_data if not image_number else image_data[image_number - 1]

    raise AttributeError("No acquisition matches the supplied series description")


def acquisition_by_series_description(
    series_description: str, parsed_input: list
) -> DicomSeriesList:
    """
    retrieve all datasets from a serie
    :param series_description:
    :param parsed_input: list of DicomSeriesLists
    :return: item as parsed by the wadwrapper lib
    """
    for item in parsed_input:
        if item.series_description == series_description:
            item = [x for x in item if hasattr(x, "pixel_array")]
            return item
    raise AttributeError("No acquisition matches the supplied series description")


def mask_to_coordinates(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a boolean mask to x,y coordinates
    :param mask: boolean mask representing binary image of edges
    :return: tuple (np.array(x coordinates), np.array(y coordinates))
    """
    where = np.where(mask)

    y = where[0].astype(np.float)
    x = where[1].astype(np.float)

    return np.array(x), np.array(y)


def mask_edges(
    edges: np.ndarray,
    ellipse: List,
    removal_width: int = 60,  # TODO make removal_width configurable, make sure configurable items are in MM not in px!
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove edges in order to better fit the ellipse
    Pixels (that represent an edge) are removed based on two criteria
    :param removal_width: width in pixels of the upper part of the edges that will be set to zero due to the air bubble
    :param edges: x,y coordinates representing the edges of the phantom. ([x1,x2,...], [y1, y2,...])
    :param ellipse: parameters for an ellipse as returned by fit_ellipse
    :return: tuple containing an np.array for x and an np.array for y coordinates
    """
    # cast to regular list, as np.array is immutable
    edges_x = edges[0].tolist()
    edges_y = edges[1].tolist()

    center_x = ellipse[0]
    center_y = ellipse[1]

    edges_no_top_x, edges_no_top_y = _remove_top(
        edges_x, edges_y, removal_width, center_x, center_y
    )

    return np.array(edges_no_top_x), np.array(edges_no_top_y)


def _remove_top(
    x_coordinates: List, y_coordinates: List, removal_width: int, center_x, center_y
) -> Tuple[List, List]:
    """
    Remove top edges that are above center Y and between minus and plus half removal width of X
    :param center: x,y coordinates for the center of the ellipse
    :param edges: ([x1, x2,...], [y1, y2,...]) --> must be normal lists, because np.array is immutable
    :param removal_width: total width in pixels for removal. half of this will be on each side of center X
    :return:
    """

    half_removal_width = removal_width / 2
    removal_min = center_x - half_removal_width
    removal_max = center_x + half_removal_width

    indices_to_remove = []  # cannot remove while iterating, so keep track of indices
    for index, value in enumerate(x_coordinates):
        if (
            removal_min < value < removal_max and y_coordinates[index] < center_y
        ):  # y coordinates are reversed?
            indices_to_remove.append(index)

    indices_to_remove.sort(
        reverse=True
    )  # removing by index must be done from high to low to prevent index errors
    for index in indices_to_remove:
        del x_coordinates[index]
        del y_coordinates[index]

    return x_coordinates, y_coordinates


def detect_edges(image, sigma=0.3, low_threshold=750, high_threshold=800) -> np.ndarray:
    """
    Detect edges on a 2d array
    :param high_threshold: high threshold for the hysteresis thresholding
    :param low_threshold: low threshold for the hysteresis thresholding
    :param sigma: width of the Gaussian
    :param image: 2d numpy array
    :return binary array of same dimensions as numpy_array representing the detected edges
    """

    # canny requires floats
    edges = feature.canny(
        image.astype("float32"),
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    return edges


def radon_transform(image: np.ndarray, max_deg: float = 180.0) -> np.ndarray:
    """Generate a sinogram for an image
    :param image: 2 dimensional data
    :param max_deg: maximum projection angle
    :return:
    """
    theta = np.linspace(0.0, max_deg, max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    return sinogram


def plot_sinogram(sinogram, range=180.0) -> None:
    dx, dy = 0.5 * range / max(sinogram.shape), 0.5 / sinogram.shape[0]
    plt.imshow(
        sinogram,
        cmap=plt.cm.gist_heat_r,
        extent=(0, 180, -sinogram.shape[0] / 2, sinogram.shape[0] / 2),
        aspect="auto",
    )
    plt.show(block=True)


def param_or_default(params: dict, setting: str, default: Any) -> Any:
    if setting in params:
        return params[setting]
    else:
        return default


def retrieve_ellipse_parameters(image_data, mask_air_bubble=True):
    """

    :param mask_air_bubble: disable masking of the air bubble
    :param image_data: np.array
    :return: [major axis length, minor axis length, center x coordinate, center y coordinate, angle of rotation]
    """
    # obtain a binary image (mask) representing the edges on the image
    edges = detect_edges(image_data)
    # convert the mask to coordinates
    edge_coordinates = mask_to_coordinates(edges)
    # do a preliminary fitting of the ellipse
    # ellipse = fit_ellipse(edge_coordinates[0], edge_coordinates[1])
    from skimage.measure import EllipseModel

    # set coordinates to format that EllipseModel expects
    xy = np.array(
        [
            [edge_coordinates[0][idx], edge_coordinates[1][idx]]
            for idx in range(len(edge_coordinates[0]))
        ]
    )
    ellipse_model = EllipseModel()
    if ellipse_model.estimate(xy):
        ellipse = ellipse_model.params
        # TODO else raise
    if mask_air_bubble:
        # create a new mask using the preliminary fit to remove air-bubble
        edge_coordinates_masked = mask_edges(edge_coordinates, ellipse)
        xy = np.array(
            [
                [edge_coordinates_masked[0][idx], edge_coordinates_masked[1][idx]]
                for idx in range(len(edge_coordinates_masked[0]))
            ]
        )
        ellipse_model = EllipseModel()
        if ellipse_model.estimate(xy):
            ellipse = ellipse_model.params
            # TODO else raise
    return ellipse


def get_pixel_spacing(acquisition: DicomSeriesList) -> float:
    pixel_spacing = acquisition[0].PixelSpacing
    if pixel_spacing[0] != pixel_spacing[1]:
        raise ValueError("Unequal x,y pixel spacing not supported")
    else:
        # as they are equal, only one is needed
        pixel_spacing = pixel_spacing[0]
    return pixel_spacing


def interpolation_peak_offset(data: np.ndarray, peak_index: int) -> float:
    """Use the derivatives of the peak and its direct neighbours to find a "sub pixel" offset

    :param data: one dimensional, accessible by index, data structure
    :param peak_index: index for the peak
    :return: the approximate peak
    """
    # set x coordinates from -1 to +1 so the zero crossing can be added to the peak directly
    derived_1 = (-1, data[peak_index] - data[peak_index - 1])
    derived_2 = (1, data[peak_index + 1] - data[peak_index])

    slope = (derived_2[1] - derived_1[1]) / (derived_2[0] - derived_1[0])

    # y = mx + b --> b = y - mx
    offset = derived_1[1] - (slope * derived_1[0])
    # now solve 0 = slope*x + offset
    zero_point = -offset / slope

    return zero_point


def get_points_in_circle(
    radius: float,
    x_center: float = 0,
    y_center: float = 0,
) -> List[Tuple[int, int]]:
    """
    List all the points in a circle
    :param radius:
    :param x_center:
    :param y_center:
    :return:
    """
    x_ = np.arange(x_center - radius - 1, x_center + radius + 1, dtype=int)
    y_ = np.arange(y_center - radius - 1, y_center + radius + 1, dtype=int)
    x, y = np.where(
        (x_[:, np.newaxis] - x_center) ** 2 + (y_ - y_center) ** 2 <= radius**2
    )
    return [(x, y) for x, y in zip(x_[x], y_[y])]


def generate_ellipse() -> np.ndarray:
    """
    Generate a pronounced ellipse in a format that can be used by retrieve_ellipse_parameters.
    :return:
    """
    x0 = 115  # x center
    a = 90  # x radius
    y0 = 130  # y center
    b = 40  # y radius
    x = np.linspace(0, 512, 512)  # x values of interest
    y = np.linspace(0, 512, 512)[:, None]  # y values of interest, as a "column" array
    ellipse = ((x - x0) / a) ** 2 + (
        (y - y0) / b
    ) ** 2 <= 1  # True for points inside the ellipse
    ellipse_2 = [
        [0.0 for i in range(0, ellipse.shape[1])] for i in range(0, ellipse.shape[1])
    ]
    for idx_row, row in enumerate(ellipse):
        for idx_col, px in enumerate(row):
            if px == 0:
                ellipse_2[idx_row][idx_col] = float(random.randint(0, 5))
            else:
                ellipse_2[idx_row][idx_col] = float(random.randint(1000, 1005))
    return np.array(ellipse_2)


def get_ghosting_rois_pixel_values(
    image_data: np.ndarray,
    center_x: float,
    center_y: float,
    short_side_mm: Union[float, int],
    long_side_mm: Union[float, int],
    shift_mm: Union[float, int],
    pixel_spacing: float,
) -> (dict, List):
    """
    Retrieve the pixel values for 4 ghosting rois top, left, bottom and right of phantom roi
    :param image_data: np.array containing pixel values
    :param center_x: x center coordinate to shift from
    :param center_y: y center coordinate to shift from
    :param short_side_mm: length in mm of the ghosting roi short side
    :param long_side_mm: length in mm of the long side of the ghosting roi
    :param shift_mm: shift in mm from center
    :param pixel_spacing: pixelspacing of the image data
    :return: tuple of np array containing all pixel values and definition of the rois for plotting
    """
    # determine the center coordinates for all the ghosting rois
    bottom_roi_x = center_x
    bottom_roi_y = center_y - (shift_mm * pixel_spacing)

    left_roi_x = center_x - (shift_mm * pixel_spacing)
    left_roi_y = center_y

    top_roi_x = center_x
    top_roi_y = center_y + (shift_mm * pixel_spacing)

    right_roi_x = center_x + (shift_mm * pixel_spacing)
    right_roi_y = center_y

    bottom_values = get_pixel_values_rectangle(
        image_data,
        center_x=bottom_roi_x,
        center_y=bottom_roi_y,
        width_mm=long_side_mm,
        height_mm=short_side_mm,
        pixel_spacing=pixel_spacing,
    )

    left_values = get_pixel_values_rectangle(
        image_data,
        center_x=left_roi_x,
        center_y=left_roi_y,
        width_mm=short_side_mm,
        height_mm=long_side_mm,
        pixel_spacing=pixel_spacing,
    )

    top_values = get_pixel_values_rectangle(
        image_data,
        center_x=top_roi_x,
        center_y=top_roi_y,
        width_mm=long_side_mm,
        height_mm=short_side_mm,
        pixel_spacing=pixel_spacing,
    )

    right_values = get_pixel_values_rectangle(
        image_data,
        center_x=right_roi_x,
        center_y=right_roi_y,
        width_mm=short_side_mm,
        height_mm=long_side_mm,
        pixel_spacing=pixel_spacing,
    )

    # these are returned for plotting reasons
    rectangles = [
        [(bottom_roi_x, bottom_roi_y), long_side_mm, short_side_mm],
        [(left_roi_x, left_roi_y), short_side_mm, long_side_mm],
        [(top_roi_x, top_roi_y), long_side_mm, short_side_mm],
        [(right_roi_x, right_roi_y), short_side_mm, long_side_mm],
    ]

    return (
        {
            "bottom": np.array(bottom_values),
            "left": np.array(left_values),
            "top": np.array(top_values),
            "right": np.array(right_values),
        },
        rectangles,
    )


def get_background_rois_pixel_values(
    image_data: np.ndarray,
    center_x: float,
    center_y: float,
    sides_mm: Union[float, int],
    shift_mm: Union[float, int],
    pixel_spacing: float,
) -> (np.ndarray, List):
    """
    Retrieve the pixel values for 4 background rois in every corner of the image
    :param image_data: np.array containing pixel values
    :param center_x: x center coordinate to shift from
    :param center_y: y center coordinate to shift from
    :param sides_mm: length in mm of the background roi sides
    :param shift_mm: shift in mm from center
    :param pixel_spacing: pixelspacing of the image data
    :return:
    """

    # determine the center coordinates for all the background rois
    lower_left_roi_x = center_x - (shift_mm * pixel_spacing)
    lower_left_roi_y = center_y - (shift_mm * pixel_spacing)

    upper_left_roi_x = center_x - (shift_mm * pixel_spacing)
    upper_left_roi_y = center_y + (shift_mm * pixel_spacing)

    upper_right_roi_x = center_x + (shift_mm * pixel_spacing)
    upper_right_roi_y = center_y + (shift_mm * pixel_spacing)

    lower_right_roi_x = center_x + (shift_mm * pixel_spacing)
    lower_right_roi_y = center_y - (shift_mm * pixel_spacing)

    lower_left_values = get_pixel_values_rectangle(
        image_data,
        center_x=lower_left_roi_x,
        center_y=lower_left_roi_y,
        width_mm=sides_mm,
        height_mm=sides_mm,
        pixel_spacing=pixel_spacing,
    )

    upper_left_values = get_pixel_values_rectangle(
        image_data,
        center_x=upper_left_roi_x,
        center_y=upper_left_roi_y,
        width_mm=sides_mm,
        height_mm=sides_mm,
        pixel_spacing=pixel_spacing,
    )

    upper_right_values = get_pixel_values_rectangle(
        image_data,
        center_x=upper_right_roi_x,
        center_y=upper_right_roi_y,
        width_mm=sides_mm,
        height_mm=sides_mm,
        pixel_spacing=pixel_spacing,
    )

    lower_right_values = get_pixel_values_rectangle(
        image_data,
        center_x=lower_right_roi_x,
        center_y=lower_right_roi_y,
        width_mm=sides_mm,
        height_mm=sides_mm,
        pixel_spacing=pixel_spacing,
    )
    # these are returned for plotting reasons
    rectangles = [
        [(lower_left_roi_x, lower_left_roi_y), sides_mm, sides_mm],
        [(upper_left_roi_x, upper_left_roi_y), sides_mm, sides_mm],
        [(upper_right_roi_x, upper_right_roi_y), sides_mm, sides_mm],
        [(lower_right_roi_x, lower_right_roi_y), sides_mm, sides_mm],
    ]

    return (
        np.concatenate(
            (
                lower_left_values,
                upper_left_values,
                upper_right_values,
                lower_right_values,
            ),
            axis=None,
        ),
        rectangles,
    )


def get_pixel_values_circle(
    image_data: np.ndarray,
    center_x: Union[int, float],
    center_y: Union[int, float],
    diameter_mm: Union[int, float],
    pixel_spacing: Union[int, float],
) -> np.ndarray:
    """
    Return the pixel values within a circle placed on an image
    Only image data with equal pixelspacing for x and y supported
    :param pixel_spacing: distance of pixels in mm
    :param image_data: np.array containing pixel values
    :param center_x: x coordinate for the center of the circle
    :param center_y: y coordinate for the center of the circle
    :param diameter_mm: diameter of the circle in  mm
    :return: np.array containing the (unsorted) values of the pixels within the circle
    """

    center = np.array([center_x, center_y])
    radius = diameter_mm / 2
    pixel_values = []
    for row_index, row in enumerate(image_data):
        for pixel_index, pixel_value in enumerate(row):
            current = np.array([row_index, pixel_index])
            if np.linalg.norm(center - current) * pixel_spacing < radius:
                pixel_values.append(pixel_value)
    return np.array(pixel_values)


def get_pixel_values_rectangle(
    image_data: np.ndarray,
    center_x: Union[int, float],
    center_y: Union[int, float],
    width_mm: Union[int, float],
    height_mm: Union[int, float],
    pixel_spacing: Union[int, float],
) -> np.array:
    """
    Only image data with equal pixelspacing for x and y supported
    :param pixel_spacing: distance of pixels in mm
    :param image_data:  np.array containing pixel values
    :param center_x: x coordinate for the center of the rectangle
    :param center_y: y coordinate for the center of the rectangle
    :param width_mm: width in mm of the rectangle of which the containing pixel values should be obtained
    :param height_mm: height in mm of the rectangle of which the containing pixel values should be obtained
    :return: np.array containing the (unsorted) values of the pixels within th rectangle
    """
    half_width = width_mm / 2
    half_height = height_mm / 2
    # need integers to access the image data
    low_x = round(center_x - half_width * pixel_spacing)
    low_y = round(center_y - half_height * pixel_spacing)
    high_x = round(center_x + half_width * pixel_spacing)
    high_y = round(center_y + half_height * pixel_spacing)

    pixel_values = image_data[low_y : high_y + 1, low_x : high_x + 1]
    return np.array(pixel_values)


def get_coordinates_relative_to_center(
    distance_mm: Union[int, float],
    height_difference_mm: Union[int, float],
    center_x: Union[int, float],
    center_y: Union[int, float],
    pixel_spacing_mm: Union[int, float] = 1,
) -> Tuple[float, float]:
    """
    Calculate coordinates that are relative to the center by giving the distance to center and the difference in height
    in cm.
    left of center is negative distance, below center is negative height
    only square pixel spacing supported (x,y are equal)
    :param pixel_spacing_mm: coordinate to distance ratio; optional
    :param distance_mm: center to point distance in cm
    :param height_difference_mm: height (y axis) difference of center and point, relative to the point
    :param center_x: x coordinate of center
    :param center_y: y coordinate of center
    :return: returns a tuple of floats (x,y)
    """

    new_y = center_y - (height_difference_mm / pixel_spacing_mm)
    if distance_mm in [0, 0.0]:
        return center_x, new_y

    # inverse pythagoras, correct for pixel spacing
    x_delta = np.sqrt(distance_mm**2 - height_difference_mm**2) / pixel_spacing_mm

    if distance_mm < 0:
        new_x = center_x - x_delta
    else:
        new_x = center_x + x_delta

    return new_x, new_y


def elliptical_mask(
    radius_x: Union[float, int],
    radius_y: Union[float, int],
    center_x: Union[float, int],
    center_y: Union[float, int],
    dimension_x: int,
    dimension_y: int,
    angle: Union[float, int] = 0,
    ignore_top: float = None,
) -> np.ndarray:
    """
    Create a mask based on an ellipse.

    :param radius_x: radius for the x axis
    :param radius_y: radius for the y axis
    :param center_x: Center coordinate for the x axis
    :param center_y: Center coordinate for the y axis
    :param angle: angle of rotation for the bounding box of the ellipse
    :param dimension_x: x axis size of the mask
    :param dimension_y: y axis size of the mask
    :param ignore_top: cut of a percentage of the top of the ellipse
    :return:
    """

    mask = np.zeros((dimension_x, dimension_y), dtype=np.float)
    rr, cc = ellipse(
        center_x, center_y, radius_x, radius_y, shape=mask.shape, rotation=angle
    )
    if ignore_top:
        # ignore
        rr_size = rr.max() - rr.min()
        new_rr_size = rr_size * ignore_top
        new_rr_max = rr.max() - new_rr_size
        rr[rr < new_rr_max] = -1
        new_rr, new_cc = [], []
        # strip the -1s, as rr,cc are indicies for coordinates, a value indicates a value at said coordinate
        for idx, val in enumerate(rr):
            if val >= 0:
                new_rr.append(val)
                new_cc.append(cc[idx])
        rr, cc = np.array(new_rr), np.array(new_cc)
    mask[rr, cc] = 1

    return mask
