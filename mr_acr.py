#!/usr/bin/env python3
import logging

from util import parse_inputs
from wad_qc.module import pyWADinput as wad_input

from actions import acqdatetime
from actions import resonance_frequency
from actions import rf_transmitter_amplitude
from actions import signal_noise_ratio
from actions import ghosting
from actions import image_uniformity
from actions import geometry
from actions import shim

logger = logging.getLogger(__name__)

available_actions = {
    'acqdatetime': acqdatetime,
    'resonance_frequency': resonance_frequency,
    'rf_transmitter_amplitude': rf_transmitter_amplitude,
    'signal_noise_ratio': signal_noise_ratio,
    'ghosting': ghosting,
    'image_uniformity': image_uniformity,
    'geometry': geometry,
    'shim': shim
}


if __name__ == "__main__":
    data, results, config = wad_input()
    parsed_inputs = parse_inputs(data.series_filelist)

    for action, configuration in config['actions'].items():
        if action in available_actions:
            try:
                available_actions[action](parsed_inputs, results, configuration)
            except Exception as e:
                logger.warning(f"'{action}' failed, skipping...")
                logger.warning(f"Exception text:" + str(e))

            except:
                logger.warning(f"'{action}' failed, skipping...")
        else:
            logger.warning(f"'{action}' not available, skipping...")

    results.write()

"""
TODO:
SHim --> refactor in vendor & generic part, plot
Results --> verify addresult usage. graphics etc
Config/params/descriptions
validate
"""
