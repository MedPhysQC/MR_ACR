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
Implementation of reading DICOM images containing the scanned B0-map data,
typically a dual-echo gradient echo. 

Output is an instance of the B0_map class with magnitude and phase images and
some meta-data (data type, delta TE, field strength).

TODO: Implement SiemensServiceStimEcho
  Output would simply be a copy of the input data image to allow the user
  to count the number of lines. Additional output could be the TE and the
  spacing between lines in PPM

Changelog:
    20240919: initial version 

"""

__version__ = '20230919'
__author__ = 'jkuijer'

from enum import Enum
import numpy as np

from util import (
    param_or_default,
    series_by_series_description,
    image_data_from_series,
)


class B0_DataType(Enum):
    B0_PHASE_RAD = 1
    B0_PPM = 2
    B0_HZ = 3

class B0_Map:
    def __init__(
            self, 
            image_data_mag = None, 
            image_data_phs = None, 
            data_type: B0_DataType = None, 
            delta_TE = None,
            field_strength_T = None
        ):
        self.image_data = (image_data_mag, image_data_phs)
        self.data_type = data_type
        self.delta_TE = delta_TE
        self.field_strength_T = field_strength_T




def GE_VUMC_custom(series_description, parsed_input, action_config):
    print("  getting B0 series for GE_VUMC_custom")
    
    # GE has one single series that has magnitude images first, then phase images
    print( "  search for configured SeriesDescription: " + series_description )

    # configuration of images numbers for magnitude and phase images
    # if not configured assume single slice B0 map with image numbers 1 and 2
    image_number_mag = int(
        param_or_default(action_config["params"], "B0_uniformity_image_number_magnitude", 1)
    )
    image_number_phs = int(
        param_or_default(action_config["params"], "B0_uniformity_image_number_phase", 2)
    )
    print( f"  image numbers magnitude = {image_number_mag}, phase = {image_number_phs}" )

    # find series
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
    
    # check sequence name of VUmc custom B0 map
    if ['0x0019', '0x109c'] in series[0]:
        seq_name = 'fgre_B0'
        print( "  check if ['0x0019', '0x109c'] contains \'" + seq_name + "\'" )
        if str(series[0]['0x0019', '0x109c'].value).find(seq_name) != -1:
            print( "  detected VUmc custom B0 map for GE." )
        else:
            print( "  could not detect VUmc custom B0 map for GE." )
            raise ValueError("Images not recognized as VUmc custom B0 map for GE.")
    else:
        print( "  could not detect VUmc custom B0 map for GE." )
        raise ValueError("Images not recognized as VUmc custom B0 map for GE.")
        


    # get the pixel data
    image_data_mag = image_data_from_series(series, image_number = image_number_mag).astype("float32")
    image_data_phs = image_data_from_series(series, image_number = image_number_phs).astype("float32")

    # For GE/VUmc custom, the EchoTime in DICOM header is actually a delta-TE of the two echoes
    delta_TE = float( series[0].EchoTime )
    # conversion phase map from pixel values to radians
    factor = 1.0 / 1000
    offset = 0

    # convert from phase image from pixel values to radians
    image_data_phs_rad = image_data_phs * factor - offset

    field_strength_T = float( series[0].MagneticFieldStrength )
    
    return B0_Map( 
        image_data_mag = image_data_mag,
        image_data_phs = image_data_phs_rad, 
        data_type = B0_DataType.B0_PHASE_RAD,
        delta_TE = delta_TE,
        field_strength_T = field_strength_T
    )
    

def GE_B0map(series_description, parsed_input, action_config):
    print("  getting B0 series for GE_B0map")
    
    # GE has one single series that first magnitude images first, then phase images
    print( "  search for configured SeriesDescription: " + series_description )

    # configuration of images numbers for magnitude and phase images
    # must be single slice B0 map with phase in #1 and magn in #2
    image_number_mag = 2
    image_number_phs = 1
    print( f"  image numbers magnitude = {image_number_mag}, phase = {image_number_phs}" )

    # find series
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
    
    # check sequence name of VUmc custom B0 map
    if ['0x0019', '0x109c'] in series[0]:
        seq_name = 'B0map'
        print( "  check if ['0x0019', '0x109c'] contains \'" + seq_name + "\'" )
        if str(series[0]['0x0019', '0x109c'].value).find(seq_name) != -1:
            print( "  detected research type-in B0 map for GE." )
        else:
            print( "  could not detect research type-in B0 map for GE. Name not identical to \'B0map\'" )
            raise ValueError("Images not recognized as B0 map for GE.")
    else:
        print( "  could not detect research type-in B0 map for GE." )
        raise ValueError("Images not recognized as B0 map for GE.")
        
    # check number of images in the B0 map series, should have only two
    if len( series ) > 2:
        print( "  research type-in B0 map for GE should have only 2 images." )
        raise ValueError("More than 2 images in GE B0 map series.")
        

    # get the pixel data
    image_data_mag = image_data_from_series(series, image_number = image_number_mag).astype("float32")
    image_data_phs = image_data_from_series(series, image_number = image_number_phs).astype("float32")

    # For GE/VUmc custom, the EchoTime in DICOM header is actually a delta-TE of the two echoes
    delta_TE = float( series[0].EchoTime )
    # conversion phase map from pixel values to radians
    factor = 1.0 / 1000
    offset = 0

    # convert from phase image from pixel values to radians
    image_data_phs_rad = image_data_phs * factor - offset
    
    field_strength_T = float( series[0].MagneticFieldStrength )

    return B0_Map( 
        image_data_mag = image_data_mag,
        image_data_phs = image_data_phs_rad, 
        data_type = B0_DataType.B0_PHASE_RAD,
        delta_TE = delta_TE,
        field_strength_T = field_strength_T
    )


def Philips_B0map(series_description, parsed_input, action_config):
    raise NotImplementedError("Philips_B0map not yet implemented")
    B0_map = None
    return B0_map


def PhilipsDoubleEcho(series_description, parsed_input, action_config):
    print("  **** TODO: test implementation for PhilipsDoubleEcho ****")
    print("  getting B0 series for PhilipsDoubleEcho")
    
    # Philips has one single series with two magnitude images first, then two phase images
    print( "  search for configured SeriesDescription: " + series_description )

    # find series
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

    # check if there are 4 images
    if len(series) != 4:
        raise ValueError(f"SHIM: Expected 4 images, found {len(series)}")

    # magnitude index = 0; used for determining center of the phantom
    series_mag = series[0]

    # check if series has FFE in 2001,1020
    private_2001_1020 = series_mag[0x20011020].value.strip()
    if private_2001_1020 != "FFE":
        raise ValueError(
            f"SHIM: Could not detect Philips double echo FFE. Expected 'FFE'; got {private_2001_1020}"
        )

    # phase1 index = 2, phase2 index=3
    series_phs1 = series[2]
    series_phs2 = series[3]

    # convert pixel values to radians
    if hasattr(series_phs1, "RescaleSlope") and hasattr(series_phs2, "RescaleSlope"):
        factor1 = series_phs1.RescaleSlope / 1000.0
        offset1 = series_phs1.RescaleIntercept / 1000.0
        factor2 = series_phs2.RescaleSlope / 1000.0
        offset2 = series_phs2.RescaleIntercept / 1000.0
    elif (
        hasattr(series_phs1, "RealWorldValueMappingSequence")
        and hasattr(series_phs1.RealWorldValueMappingSequence, "Item_1")
        and hasattr(series_phs1.RealWorldValueMappingSequence.Item_1, "RealWorldValueSlope")
        and hasattr(
            series_phs1.RealWorldValueMappingSequence.Item_1,
            "RealWorldValueIntercept",
        )
        and hasattr(series_phs2, "RealWorldValueMappingSequence")
        and hasattr(series_phs2.RealWorldValueMappingSequence, "Item_1")
        and hasattr(series_phs2.RealWorldValueMappingSequence.Item_1, "RealWorldValueSlope")
        and hasattr(
            series_phs2.RealWorldValueMappingSequence.Item_1,
            "RealWorldValueIntercept",
        )
    ):
        factor1 = (
            series_phs1.RealWorldValueMappingSequence.Item_1.RealWorldValueSlope / 1000.0
        )
        offset1 = (
            series_phs1.RealWorldValueMappingSequence.Item_1.RealWorldValueIntercept
            / 1000.0
        )
        factor2 = (
            series_phs2.RealWorldValueMappingSequence.Item_1.RealWorldValueSlope / 1000.0
        )
        offset2 = (
            series_phs2.RealWorldValueMappingSequence.Item_1.RealWorldValueIntercept
            / 1000.0
        )
    else:
        raise AttributeError(
            "SHIM: RescaleSlope or RescaleIntercept could not be determined. Shim analysis skipped"
        )

    image_data_mag  = image_data_from_series(series, image_number = 1).astype("float32")
    image_data_phs1 = image_data_from_series(series, image_number = 3).astype("float32")
    image_data_phs2 = image_data_from_series(series, image_number = 4).astype("float32")
    
    # convert from phase image from pixel values to radians
    image_data_phs1_rad = image_data_phs1 * factor1 - offset1
    image_data_phs2_rad = image_data_phs2 * factor2 - offset2

    delta_phase_rad = image_data_phs2_rad - image_data_phs1_rad
    delta_TE = series_phs2.EchoTime - series_phs1.EchoTime

    field_strength_T = float( series[0].MagneticFieldStrength )
    
    return B0_Map( 
        image_data_mag = image_data_mag,
        image_data_phs = delta_phase_rad, 
        data_type = B0_DataType.B0_PHASE_RAD,
        delta_TE = delta_TE,
        field_strength_T = field_strength_T
    )


def SiemensPhaseDifference(series_description, parsed_input, action_config):
    print("  getting B0 series for SiemensPhaseDifference")
    
    # Siemens has two sequential series, first magnitude, second phase image
    print( "  search for configured SeriesDescription: " + series_description )

    # Check if next series has same SeriesDescription
    series_mag = None
    series_phs = None
    for item in parsed_input:
        if (item.series_description == series_description):
            item = [x for x in item if hasattr(x, "pixel_array")]
            if not series_mag:
                series_mag = item
            else:
                if not series_phs:
                    series_phs = item

    if series_mag:
        print(
            "  matched with series number: "
            + str( series_mag[0]["SeriesNumber"].value )
            + "\n  study_instance_uid: '"
            + series_mag[0]["StudyInstanceUID"].value
            + "'\n  series_instance_uid: '"
            + series_mag[0]["SeriesInstanceUID"].value
            + "'"
        )
    else:
        raise AttributeError("No match with the supplied series description")
        
    if series_phs:
        print(
            "  found phase series with series number: "
            + str( series_phs[0]["SeriesNumber"].value )
            + "'\n  series_instance_uid: '"
            + series_phs[0]["SeriesInstanceUID"].value
            + "'"
        )
    else:
        raise AttributeError("Could not find matching phase series")

    # check if ImageType has attribute P (for Phase)
    if not 'P' in series_phs[0].ImageType:
        raise AttributeError("Phase image does not have ImageType P")

    # get the pixel data
    image_data_mag = image_data_from_series(series_mag).astype("float32")
    image_data_phs = image_data_from_series(series_phs).astype("float32")

    if len( series_phs ) == 2:
        # need to calculate phase difference from the two phase images
        image_data_phs = image_data_from_series(series_phs, 2).astype("float32") - image_data_phs
        # delta TE from DICOM headers
        delta_TE = float(series_phs[1].EchoTime) - float(series_phs[0].EchoTime)
        # conversion phase map from pixel values to radians
        factor = 2 * np.pi / 4096
        offset = 0
    else:
        # For phase difference image: Siemens doesn't store the delta TE in the
        # DICOM header but it is fixed in the product sequence
        delta_TE = 4.76
        # conversion phase map from pixel values to radians
        factor = 2 * np.pi / 4096
        offset = np.pi

    # convert from phase image from pixel values to radians
    image_data_phs_rad = image_data_phs * factor - offset
    
    field_strength_T = float( series_mag[0].MagneticFieldStrength )

    return B0_Map(
        image_data_mag = image_data_mag,
        image_data_phs = image_data_phs_rad, 
        data_type = B0_DataType.B0_PHASE_RAD,
        delta_TE = delta_TE,
        field_strength_T = field_strength_T
    )


def SiemensServiceStimEcho(series_description, parsed_input, action_config):
    raise NotImplementedError("SiemensServiceStimEcho not yet implemented")
    B0_map = None
    return B0_map
