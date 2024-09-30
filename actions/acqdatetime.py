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

from util import DicomSeriesList, series_by_series_description
from wad_qc.modulelibs import wadwrapper_lib as wad_lib


def acqdatetime(parsed_input: List[DicomSeriesList], result, action_config) -> None:
    """

    :param parsed_input:
    :param result:
    :param action_config:
    """
    actionName = "acqdatetime"
    print( "> action " + actionName )
    level = action_config["params"]["datetime_level"]
    series_description = action_config["params"]["datetime_series_description"]
    print( "  search for configured SeriesDescription: " + series_description )
    
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

    dt = wad_lib.get_datetime(ds=series[0], datetime_level=level)
    result.addDateTime("AcquisitionDateTime", dt)
    
    if "SoftwareVersions" in series[0]:
        if type(series[0].SoftwareVersions) is str:
            version_string = series[0].SoftwareVersions
        else:
            version_string = ' '.join(series[0].SoftwareVersions)
        print( "  scanner software version = \'" + version_string + "\'" )
        result.addString( "Scanner_software_version", version_string ) 
