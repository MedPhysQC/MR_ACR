#!/usr/bin/env python3

# Should file be written to tmp dir for inspection? 0 = no, 1 = yes
g_WriteInputFilesToTempDir = 0


def GetCommandLineArguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Dicom folder", required=True)
    parser.add_argument("-r", help="Result file", required=True)
    parser.add_argument("-c", help="Config file", required=True)
    args = parser.parse_args()

    return args


def CopyInputDirectoriesToTempDir(dcmDir, configDir):
    if g_WriteInputFilesToTempDir == 0:
        return

    import os
    import tempfile
    import shutil

    sourceDcmDir = os.path.normpath(dcmDir)
    sourceDcmDir = os.path.abspath(sourceDcmDir)

    sourceConfigDir = os.path.normpath(configDir)
    sourceConfigDir = os.path.abspath(sourceConfigDir)

    destinationBaseDir = os.path.join(tempfile.gettempdir(), "tmp_MR_ACR")

    shutil.rmtree(destinationBaseDir, ignore_errors=True)
    os.makedirs(destinationBaseDir, mode=0o777, exist_ok=True)

    shutil.copytree(sourceDcmDir, os.path.join(destinationBaseDir, "dicom"))
    shutil.copytree(sourceConfigDir, os.path.join(destinationBaseDir, "config"))


def RunDockerContainer(args):
    import os

    # Dicom directory
    args.d = os.path.normpath(args.d)
    args.d = os.path.abspath(args.d)
    localDicomDir = args.d

    dockerDicomDir = os.sep + "local_dicom"

    # Result file dir and name
    args.r = os.path.normpath(args.r)
    args.r = os.path.abspath(args.r)
    localResultFileDir = os.path.dirname(args.r)
    localResultFileName = os.path.basename(args.r)

    # For the result file and images mimic the local path in the docker container, so the paths
    # are placed correctly in the result json file.
    dockerResultFileDir = localResultFileDir

    # Config file dir and name
    args.c = os.path.normpath(args.c)
    args.c = os.path.abspath(args.c)
    localConfigFileDir = os.path.dirname(args.c)
    localConfigFileName = os.path.basename(args.c)

    dockerConfigFileDir = os.sep + "local_config"

    # Prepare docker command.
    # First set base command and parameters:
    # run 		- run a container
    # --rm 		- Remove container instance after use

    command = "docker run --rm "

    # Add bind mounts to map local files
    command = command + f"-v {localDicomDir}:{dockerDicomDir} "
    command = command + f"-v {localResultFileDir}:{dockerResultFileDir} "
    command = command + f"-v {localConfigFileDir}:{dockerConfigFileDir} "

    # Add container name
    command = command + "mr_acr "

    # Add python command to run
    command = command + f"pipenv run python {os.sep}mr_acr{os.sep}mr_acr.py "

    # Add processed parameters
    command = command + f"-r {dockerResultFileDir + os.sep + localResultFileName} "
    command = command + f"-c {dockerConfigFileDir + os.sep + localConfigFileName}  "
    command = command + f"-d {dockerDicomDir} "

    # Copy file to tmp dir for inspection if required
    CopyInputDirectoriesToTempDir(localDicomDir, localConfigFileDir)

    commandResult = os.system(command)
    return commandResult


if __name__ == "__main__":
    args = GetCommandLineArguments()
    RunDockerContainer(args)
