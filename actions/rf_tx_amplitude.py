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
Get transmitter amplitude from the DICOM header

Workflow:
input: any image of the QC test
output: transmitter amplitude
       

Changelog:
    20240919: initial version
    20241213 / jkuijer: improved support for Siemens CSA protocol reading
    - support for CSA protocol in 0x1020
    - get encoding type (latin, utf etc) from the header and use it to read CSA protocol
"""

__version__ = '20241213'
__author__ = 'jkuijer'


from typing import List
import numbers

from util import DicomSeriesList, series_by_series_description

def rf_transmitter_amplitude(
    parsed_input: List[DicomSeriesList], result, action_config
):
    """
    Can sometimes be found in a private DICOM header
    Not implemented as NKI-Avl does not have scanners that include this information in DICOM headers
    :param parsed_input: List of DicomSeriesList objects.
    :param result: Result object as provided by wad wq
    :param action_config: action_config object
    :return:
    """

    # import time
    # time.sleep(10)

    actionName = "rf_transmitter_amplitude"
    print( "> action " + actionName )
    rf_tag = action_config["params"]["rf_transmitter_amplitude_tag"]
    series_description = action_config["params"]["rf_transmitter_series_description"]
    series = series_by_series_description(series_description, parsed_input)
    print( "  search for configured SeriesDescription: " + series_description )

    print(
        "  matched with series number: "
        + str( series[0]["SeriesNumber"].value )
        + "\n  study_instance_uid: '"
        + series[0]["StudyInstanceUID"].value
        + "'\n  series_instance_uid: '"
        + series[0]["SeriesInstanceUID"].value
        + "'"
    )


    try:
        # DICOM field tag in hex format, e.g. "0x0019, 0x1094"
        # get the two hex numbers as list, split at comma
        rf_tag_nospc = rf_tag.replace(" ", "")
        rf_tag_parsed = list( map( str, rf_tag_nospc.split(",") ) )
    except ValueError:
        raise ValueError( f"Error: incorrect format of configured tag: {rf_tag}" )

    # we should have two numbers
    if len(rf_tag_parsed) != 2:
        raise ValueError( f"Error: incorrect format of configured tag: {rf_tag}" )
        

    TxAmpl = None
    try:
        if rf_tag_parsed[1] == '0x1019' or rf_tag_parsed[1] == '0x1020':
            # assume it is Siemens protocol in private field    
            print( f"  assuming Siemens private field in {rf_tag_parsed[1]}")
            encoding_type = series[0].read_encoding[0]
            print( f"  encoding type: {encoding_type}" )
            protocol = series[0][rf_tag_parsed].value.decode( encoding_type )
            
            # decode the substring flReferenceAmplitude\t = \t374.940124512\n
            refAmplTxtIndex = protocol.find( "flReferenceAmplitude" )
            equalSignTxtIndex = protocol.find( "=", refAmplTxtIndex )
            newLineTxtIndex = protocol.find( "\n", equalSignTxtIndex )
            TxAmpl = float( protocol[equalSignTxtIndex+1:newLineTxtIndex] )
        else:
            # try to get that value as numerical content
            tx_field = series[0][rf_tag_parsed].value
            if type(tx_field) is str:
                print( f"  try to read TX ampl as string from {rf_tag}")
                TxAmpl = float(tx_field)
            else:
                if isinstance(tx_field, numbers.Number):
                    print( f"  try to read TX ampl as number from {rf_tag}")
                    TxAmpl = float(tx_field)
                else:
                    print( f"  tag {rf_tag} has unknown data type {type(tx_field)}")
            
    except ValueError:
        raise ValueError(
            f"Cannot determine RF Transmitter amplitude from tag {rf_tag}"
        )

    if TxAmpl:
        result.addFloat("RF_TransmitterAmplitude", TxAmpl)
        print( "  RF transmitter amplitude = {:0.2f}".format(TxAmpl))
