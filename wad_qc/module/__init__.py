from wad_qc.module.moduledata import ModuleData
from wad_qc.module.moduleresults import ModuleResults
import json
import jsmin


def pyWADinput():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Dicom folder', required=True)
    parser.add_argument('-r', help='Result file', required=True)
    parser.add_argument('-c', help='Config file', required=True)
    args = parser.parse_args()

    data = ModuleData(args.d)
    with open(args.c) as f:
        validjson = jsmin.jsmin(f.read())  # strip comments and stuff from more readable json
    try:
        config = json.loads(validjson)
    except json.decoder.JSONDecodeError:
        print("Error: config file is not in valid json format!")
        exit(False)

    results = ModuleResults(args.r)

    return data, results, config
