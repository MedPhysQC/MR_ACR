from typing import List

from util import DicomSeriesList


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
    return
    rf_tag = action_config["params"]["rf_transmitter_amplitude_tag"]
    series_description = action_config["params"]["rf_transmitter_series_description"]
    acquisition = acquisition_by_series_description(series_description, parsed_input)

    actionName = "rf_transmitter_amplitude"
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

    try:
        # check if hex is valid; must be hex as this is stored in a private tag
        rf_tag_parsed = [int(tag, 16) for tag in rf_tag]
    except ValueError:
        raise ValueError(
            f"Cannot determine RF Transmitter amplitude because of incorrect tag value ({rf_tag})"
        )

    result.addFloat(
        "RfTransmitterAmplitude", acquisition[0][rf_tag_parsed[0], rf_tag_parsed[1]]
    )
