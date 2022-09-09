from typing import List

from util import DicomSeriesList, acquisition_by_series_description
from wad_qc.modulelibs import wadwrapper_lib as wad_lib


def acqdatetime(parsed_input: List[DicomSeriesList], result, action_config) -> None:
    """

    :param parsed_input:
    :param result:
    :param action_config:
    """
    level = action_config["params"]["datetime_level"]
    series_description = action_config["params"]["datetime_series_description"]
    acquisition = acquisition_by_series_description(series_description, parsed_input)
    dt = wad_lib.get_datetime(ds=acquisition[0], datetime_level=level)

    result.addDateTime("AcquisitionDateTime", dt)
