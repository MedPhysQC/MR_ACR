#!/usr/bin/env python
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
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
# The analysis modules can be found on
# https://github.com/MedPhysQC


"""
Analysis of MRI images of the ACR phantom

mr_acr is an analysis module for QC of MRI scanners. The intended use is 
anaysis of images of the ACR phantom. Metrics include geometry (phantom
diameter and length), SNR, ghosting, image uniformity and B0 homogeneity. The
cofiguration options may allow the module to be used with other types of
phantoms.

input: study containing various series targeted at tests implemented in
       sub-modules.

Functionality is implemented in sub-modules in the "actions" directory. The
config file lists the actions to be executed accompanied by the criteria for
a matching series or image. Each configured action is executed once. Series and
image selection is implemented in the sub-module.

Limitations
Generally, the first series that matched the filter criteria is used for input.
Consequently, if multiple versions of test data series exist (e.g. multiple SNR
tests with various coils) then only a single series will be analysed.


Changelog:
    20240919: initial version of the Python "factory module", revised by
              J Kuijer (AmsterdamUMC), based on development versions by
              M Barsingerhorn (NKI) and G Millenaar (NKI). This Python module
              is a re-implementation of WADlab2_MR written in Matlab.
"""

__version__ = '20230919'
__author__ = 'jkuijer, mbarsingerhorn, gmillenaar'

import logging

from util import parse_inputs
from wad_qc.module import pyWADinput as wad_input

from actions import acqdatetime
from actions import resonance_frequency
from actions import rf_transmitter_amplitude
from actions import signal_noise_ratio
from actions import signal_noise_ratio_uncombined
from actions import ghosting
from actions import image_uniformity
from actions import geometry_xy
from actions import geometry_z
from actions import B0_uniformity

logger = logging.getLogger(__name__)

available_actions = {
    'acqdatetime': acqdatetime,
    'resonance_frequency': resonance_frequency,
    'rf_transmitter_amplitude': rf_transmitter_amplitude,
    'signal_noise_ratio': signal_noise_ratio,
    'signal_noise_ratio_uncombined': signal_noise_ratio_uncombined,
    'ghosting': ghosting,
    'image_uniformity': image_uniformity,
    'geometry_xy': geometry_xy,
    'geometry_z': geometry_z,
    'B0_uniformity': B0_uniformity
}


if __name__ == "__main__":
    data, results, config = wad_input()
    parsed_inputs = parse_inputs(data.series_filelist)

    for action, configuration in config['actions'].items():
        if action in available_actions:
            try:
                available_actions[action](parsed_inputs, results, configuration)
            except Exception as e:
                logger.warning(f"ERROR: '{action}' failed, skipping...")
                logger.warning("Exception text:" + str(e))

            except:
                logger.warning(f"ERROR: '{action}' failed, skipping...")
        else:
            logger.warning(f"WARNING: configured action '{action}' is not available, skipping...")

    results.write()
