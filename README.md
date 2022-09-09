# MR_ACR

## Repository for the MR_ACR module for WAD-QC2.0


## Overview

The main python file must be executable and expects 3 arguments:

- -c <config_file>
- -d <study_data_folder>
- -r <results_file>

These arguments are parsed using the provided libraries.

### wad_qc directory
The wad_qc directory is included to allow the plugin to be run using the provided libraries
for input parsing without having the need to install the wad_qc framework. 
This directory is not to be included when packaging the plugin for use in wad_qc.

### legacy_config directory
For convenience the wad 1.0 configuation files for internal use are included.
Will be removed once the new configs are done


### runConfigurations directory
Contains runconfigurations for use with pycharm. Runs the module like the WADqc framework.


