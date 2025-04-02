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
Analysis of image SNR. Intended use: uncombined images of phased array coils.

Workflow:
input: series with "uncombined" images of the individual coil elements of a 
       phased array coil, of at transversal "plain" slice of ACR phantom. The image should
       *NOT* be corrected for geometrical distortion by gradient non-linearity,
       as this would interpolate (or resample) the signal between pixels, thereby
       affecting the noise amplitude. Parallel imaging techniques such as
       SENSE, GRAPPA or ASSET should *NOT* be used because these affect the 
       estimation of noise in a background ROI.
output: SNR, with noise estimated from background ROIs.
       
1. Estimation of phantom centre and edge (see geometry_xy.py)
  a. calculate a combined image from the uncombined images
  b. edge detection and ellipse fit
2. Define ROIs for
  a. signal: centre of phantom
  b. noise: background signal in corners of image
3. Output is the ghosting in percent as defined in the ACR guide:
   0.655 * (signal_phantom / StdDev_background)

Changelog:
    20240919: initial version
    20250326: first stage of support for Philips uncombined images (series with additional no-RF noise-only images)
              TODO: use no-RF images for noise measurement
"""

__version__ = '20250326'
__author__ = 'jkuijer'

from typing import List

import numpy as np

from util import (
    DicomSeriesList,
    param_or_default,
    param_or_default_as_int,
    series_by_series_description,
    image_data_from_series,
    retrieve_ellipse_center,
    get_pixel_spacing,
    get_pixel_values_circle,
    get_background_rois_pixel_values,
)


def signal_noise_ratio_uncombined(parsed_input: List[DicomSeriesList], result, action_config):
    """
    Method for determining the signal to noise ratio as specified by the NVKF
    https://nvkf.nl/resources/LeidraadKwaliteitscontroleRadiologischeApparatuur3.0.pdf @ 203

    Modified version of signal_noise_ratio for uncombined coils series
    - calculates sum-of-squares combined image for ROI positioning
    - does not produce results-images

    SNR = 0.655 * Sphantom / SDbackground

    Valid if the mean / standard deviation of the background roi is 1.91

    """
    default_signal_roi_diameter_mm = 150
    default_noise_roi_sides_mm = 7
    default_background_roi_shift_mm = 108

    actionName = "signal_noise_ratio_uncombined"
    print( "> action " + actionName )

    signal_roi_diameter_mm = float(param_or_default(
        action_config["params"],
        "snr_uc_signal_roi_diameter_mm",
        default_signal_roi_diameter_mm,
    ))
    noise_roi_sides_mm = float(param_or_default(
        action_config["params"], "snr_uc_noise_roi_sides_mm", default_noise_roi_sides_mm
    ))
    background_roi_shift_mm = float(param_or_default(
        action_config["params"],
        "snr_uc_background_roi_shift_mm",
        default_background_roi_shift_mm,
    ))

    series_description = action_config["params"]["snr_uc_series_description"]
    print( "  number of series:  " + str(len(parsed_input)) )
    print( "  search for configured SeriesDescription: " + series_description )
    
    number_of_img_in_series = param_or_default_as_int(
        action_config["params"],
        "snr_uc_number_of_images_in_series",
        None
    )
    if number_of_img_in_series:
        print( "  search for configured number of images in series: " + str(number_of_img_in_series) )            
            
    setting = "snr_uc_image_number"
    params = action_config["params"]
    if setting in params:
        try:
            # allow comma-separated list of image numbers
            param = str( params[setting] )
            image_number = list( map( int, param.split(",") ) )
            print( "  use configured image number: " + str(image_number) )
        except:
            print( "  WARNING: no valid image number configured." )
            image_number = None
    else:
        print( "  no image number configured." )
        image_number = None

    
    coil_names_configured = False
    setting = "snr_uc_coil_names"
    params = action_config["params"]
    if setting in params:
        try:
            # allow comma-separated list of image numbers
            param = str( params[setting] )
            coil_names = list( map( str, param.split(",") ) )
            # regard empty string as no name configured
            if coil_names != ['']:
                print( "  use configured coil names: " + str(coil_names) )
                coil_names_configured = True
        except:
            print( "  WARNING: no valid coil names configured." )
    else:
        print( "  no coil names configured." )
    
    if not coil_names_configured: coil_names = []

    series = series_by_series_description(series_description, parsed_input, number_of_img_in_series)
    
    print(
        "  matched for series number: "
        + str( series[0]["SeriesNumber"].value )
        + "\n  study_instance_uid: '"
        + series[0]["StudyInstanceUID"].value
        + "' \n  series_instance_uid: '"
        + series[0]["SeriesInstanceUID"].value
        + "'"
    )


    # if no image was configured: loop over all images in series
    image_data_uc = []
    if image_number is None:
        image_number = range( 1, len(series)+1 )
    for i in image_number:
        image_data_uc.append( image_data_from_series( series, i ) )
        # GE needs coil names list in config
        # if coil names are not configured: try to read coil name from DICOM tag
        # Siemens
        if ( not coil_names_configured ) and [0x0021,0x114f] in series[i-1]:
            coil_names.append( series[i-1][0x0021,0x114f].value )
        else:
            # Philips
            # for uncombined series with signal images and separate noise images: 
            # - ["TemporalPositionIdentifier"] == '1' for signal image and == '2' for noise image
            # - for now just calculate SNR from background noise (ignore noise images)
            # - TODO: calc SNR using the noise images
            if (
                ( not coil_names_configured )
                and "TemporalPositionIdentifier" in series[i-1]
                # ignore noise image with ["TemporalPositionIdentifier"].value == '2'
                and series[i-1]["TemporalPositionIdentifier"].value == '1'
                and [0x2001,0x1002] in series[i-1]
            ):
                coil_names.append( series[i-1][0x2001,0x1002].value )

    comb_image_data = np.sqrt( sum( np.square( [ i.astype('float') for i in image_data_uc ] ) ) ).astype('uint16')
    
    # determine center coordinates of phantom
    [center_x, center_y] = retrieve_ellipse_center( comb_image_data )
    print(
        "  detected phantom centre coordinates: "
        + "{:.0f}".format(center_x) + " " + "{:.0f}".format(center_y)
    )

    # retrieve pixel spacing to get the pixel values from the correct positions
    pixel_spacing = get_pixel_spacing(series)

    for ( image_data, coil_name ) in zip(image_data_uc, coil_names):
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
    
        print(
            "  SNR_Coil_" + str(coil_name), "{:.2f}".format(signal_to_noise_ratio)
        )
        result.addFloat( "SNR_Coil_" + str(coil_name), "{:.2f}".format(signal_to_noise_ratio) )
