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
Analysis of B0-homogeneity.

Workflow:
input: series with B0-mapping data, typically a gradient echo with two echo times. Input it highly
       vendor-specific, therefore the reading and interpretation of the data is implemented for each
       vendor separately, and the type of B0-map data must be configured.
output: difference between max and min B0 field strength within ROI
       
1. Read B0-map data
2. Fit ellipse to phantom edge to define ROI. Relative diameter of ROI used for homogeneity is configurable.
3. Unwrap the phase images
4. Calculate phase difference image.
5. Convert phase difference to B0 in PPM
6. Calculate B0 homogeneity as difference between min and max B0 within ROI
  a. Take percentiles 0.1 and 99.9 as robust estimator of min/max B0 within the ROI
  b. Output is difference between max and min, and B0-map in PPM

Changelog:
    20240919: initial version 
"""

__version__ = '20230919'
__author__ = 'jkuijer'

from typing import List

import numpy as np
import matplotlib.pyplot as plt

import skimage.restoration


from util import (
    DicomSeriesList,
    retrieve_ellipse_parameters,
    elliptical_mask,
)

from util.get_B0_map import GE_VUMC_custom
from util.get_B0_map import GE_B0map
from util.get_B0_map import Philips_B0map
from util.get_B0_map import PhilipsDoubleEcho
from util.get_B0_map import SiemensPhaseDifference
from util.get_B0_map import SiemensServiceStimEcho

from util.get_B0_map import B0_DataType
from util.get_B0_map import B0_Map


available_B0_map_types = {
    'GE_VUMC_custom': GE_VUMC_custom,
    'GE_B0map': GE_B0map,
    'Philips_B0map': Philips_B0map,
    'PhilipsDoubleEcho': PhilipsDoubleEcho,
    'SiemensPhaseDifference': SiemensPhaseDifference,
    'SiemensServiceStimEcho': SiemensServiceStimEcho,
}




def mask_and_unwrap_image(image_data, mask, mask_phantom, unwrap=False):
    masked_image = image_data * mask_phantom
    if unwrap:
        masked_image = skimage.restoration.unwrap_phase(masked_image)
        # avoid large difference with background (zero) phase
        if np.mean( masked_image[mask] ) >  np.pi: masked_image -= 2*np.pi
        if np.mean( masked_image[mask] ) < -np.pi: masked_image += 2*np.pi
    return masked_image * mask




def B0_uniformity(parsed_input: List[DicomSeriesList], result, action_config) -> None:
    """
    B0_uniformity = 10e6 *  (max(B0) - min(B0))
    Expects a phase and a magnitude image
    The magnitude image is used to determine the center position and phantom radius.
    This is used to create a mask with a radius of 95% of the phantom,
    leaving out the upper 20% as well to account for the air bubble in the top of the phantom
    """
    actionName = "B0_uniformity"
    print( "> action " + actionName )
    series_description = action_config["params"]["B0_uniformity_series_description"]
    B0_map_type = action_config["params"]["B0_uniformity_B0_map_type"]
    roi_radius_percent = float(action_config["params"]["B0_uniformity_relative_roi_size_percent"])
    top_margin_percent = int(action_config["params"]["B0_uniformity_top_margin_percent"])


    if B0_map_type in available_B0_map_types:
        try:
            B0_map = available_B0_map_types[B0_map_type](series_description, parsed_input, action_config)
        except Exception as e:
            raise AttributeError(f"ERROR: '{B0_map_type}' failed, skipping...")
            raise AttributeError("Exception text:" + str(e))

        except:
            raise AttributeError(f"ERROR: '{B0_map_type}' failed, skipping...")
    else:
        raise AttributeError(f"WARNING: configured action '{B0_map_type}' is not available, skipping...")



    # find phantom ellipse
    # [center_x, center_y, x_axis_length, y_axis_length, phi]
    [ phantom_ellipse, _, _ ] = retrieve_ellipse_parameters(image_data=B0_map.image_data[0])

    # create large mask
    # get top 5 values avg
    # use that to determine pos or neg

    # create masks

    mask_phantom = elliptical_mask(
        center_x=phantom_ellipse[1],
        center_y=phantom_ellipse[0],
        radius_x=phantom_ellipse[3] * roi_radius_percent / 100,
        radius_y=phantom_ellipse[2] * roi_radius_percent / 100,
        angle=phantom_ellipse[4],
        dimension_x=B0_map.image_data[0].shape[0],
        dimension_y=B0_map.image_data[0].shape[1]
    )

    mask_roi = elliptical_mask(
        center_x=phantom_ellipse[1],
        center_y=phantom_ellipse[0],
        radius_x=phantom_ellipse[3] * roi_radius_percent / 100,
        radius_y=phantom_ellipse[2] * roi_radius_percent / 100,
        angle=phantom_ellipse[4],
        dimension_x=B0_map.image_data[0].shape[0],
        dimension_y=B0_map.image_data[0].shape[1],
        ignore_top=top_margin_percent / 100,
    )


    # apply mask to image and unwrap if phase data
    if B0_map.data_type is B0_DataType.B0_PHASE_RAD:
        image_data_phs_roi = mask_and_unwrap_image(
            B0_map.image_data[1], 
            mask_roi, 
            mask_phantom, 
            unwrap=True
        )
    else:
        image_data_phs_roi = mask_and_unwrap_image(
            B0_map.image_data[1], 
            mask_roi, 
            mask_phantom
        )


    gamma_rad_ms_T = 267513  # 42576 * 2 * pi;  gyromatic frequency in rad/ms * 1/T

    if B0_map.data_type is B0_DataType.B0_PHASE_RAD:
        #dB0_T = image_data_phs_roi / (gamma_rad_ms_T * B0_map.delta_TE)  # in Tesla
        #dB0_ppm = dB0_T / B0_map.field_strength_T * 1.0e6  # ppm
        dB0_ppm = image_data_phs_roi * 1.0e6 / ( gamma_rad_ms_T * B0_map.delta_TE * B0_map.field_strength_T )  # in ppm
    
    elif B0_map.data_type is B0_DataType.B0_HZ:
        raise NotImplementedError("B0_HZ not yet implemented")
    
    elif B0_map.data_type is B0_DataType.B0_PPM:
        raise NotImplementedError("B0_PPM not implemented yet")
    else:
        raise ValueError("Unknown data type in Shim analysis")


    # calculate uniformity by iterating over masked image and comparing pixel values with largest/smallest
    # use 1 and 99 percentile for more robust estimation of min/max
    dB0_ppm_masked = dB0_ppm[mask_roi]
    smallest_p = np.percentile( dB0_ppm_masked,  0.1 )
    largest_p  = np.percentile( dB0_ppm_masked, 99.9 )
    B0_uniformity_ppm = (largest_p - smallest_p)

    print( "  percentile P0.1: {:0.2f}".format(smallest_p) + " P99.9: {:0.2f}".format(largest_p) )
    print( "  B0 uniformity: {:0.2f}".format(B0_uniformity_ppm) )
    result.addFloat("B0_uniformity_ppm", B0_uniformity_ppm)

    phase_delta_filename = "unwrapped_phase_map.png"
    save_image_to_file(dB0_ppm, phase_delta_filename, "B0 uniformity [ppm]")
    result.addObject("B0_map", phase_delta_filename)


def save_image_to_file(image, name, title):
    plt.title(title)
    im = plt.imshow(image, cmap=plt.get_cmap("plasma"))
    plt.colorbar(im)

    plt.axis("off")
    plt.savefig(name, dpi=200)
    
    plt.clf()
    plt.close()
