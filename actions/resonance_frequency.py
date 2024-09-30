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

from typing import List

from util import series_by_series_description, DicomSeriesList


def resonance_frequency(parsed_input: List[DicomSeriesList], result, action_config):
    """
    dicom tag (0018,0084)
    :param action_config:
    :param result:
    :param parsed_input:
    :return:
    """
    actionName = "resonance_frequency"
    print( "> action " + actionName )
    series_description = action_config["params"][
        "resonance_frequency_series_description"
    ]
    attribute = "ImagingFrequency"
    print( "  search for configured SeriesDescription: " + series_description )

    series = series_by_series_description(
        series_description, parsed_input
    )

    print(
        "  matched with series number: "
        + str( series[0]["SeriesNumber"].value )
        + "\n  study_instance_uid: '"
        + series[0]["StudyInstanceUID"].value
        + "'\n  series_instance_uid: '"
        + series[0]["SeriesInstanceUID"].value
        + "'"
    )

    #gyromagnetic_ratio = 42.577478518
    #magnetic_field_strength = getattr(series[0], "MagneticFieldStrength")

    # Frequency in Mhz
    #expected_resonance_frequency = magnetic_field_strength * gyromagnetic_ratio
    measured_resonance_frequency = getattr(series[0], attribute)

    # Compute delta
    #delta_resonance_frequency = expected_resonance_frequency - measured_resonance_frequency

    # Convert to Hz
    #delta_resonance_frequency *= 1000000
    measured_resonance_frequency *= 1000000

    #result.addFloat("ResonanceFrequencyDelta", delta_resonance_frequency)
    result.addFloat("ResonanceFrequency_Hz", measured_resonance_frequency)
