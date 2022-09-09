from typing import List

from util import acquisition_by_series_description, DicomSeriesList


def resonance_frequency(parsed_input: List[DicomSeriesList], result, action_config):
    """
    dicom tag (0018,0084)
    :param action_config:
    :param result:
    :param parsed_input:
    :return:
    """
    series_description = action_config["params"][
        "resonance_frequency_series_description"
    ]
    attribute = "ImagingFrequency"
    try:
        acquisition = acquisition_by_series_description(
            series_description, parsed_input
        )
        result.addFloat("ResonanceFrequency", getattr(acquisition[0], attribute))
    except AttributeError:
        result.addString(
            "ResonanceFrequency",
            f"Measurement failed due to missing attribute {attribute}",
        )
