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

        actionName = "resonance_frequency"
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

        gyromagnetic_ratio = 42.577478518
        magnetic_field_strength = getattr(acquisition[0], "MagneticFieldStrength")

        # Frequency in Mhz
        expected_resonance_frequency = magnetic_field_strength * gyromagnetic_ratio
        measured_resonance_frequency = getattr(acquisition[0], attribute)

        # Compute delta
        delta_resonance_frequency = expected_resonance_frequency - measured_resonance_frequency

        # Convert to Khz
        delta_resonance_frequency *= 1000

        result.addFloat("ResonanceFrequencyDelta", delta_resonance_frequency)
    except AttributeError:
        result.addString(
            "ResonanceFrequency",
            f"Measurement failed due to missing attribute {attribute}",
        )
