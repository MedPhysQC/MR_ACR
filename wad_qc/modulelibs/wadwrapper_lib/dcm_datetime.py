"""
Functions to help finding a proper date time reflecting the moment of acquisition from dicom tags.

Changelog:
  20200508: split from wadwrapper_lib.py
"""
try:
    import pydicom as dicom
except ImportError:
    import dicom

import sys
import datetime

if sys.version_info[0] == 3:
    basestring = str


def acqdatetime_series(dcmInfile, datetime_level=None):
    """
    Read acqdatetime from dicomheaders.
    Try multiple locations and formats
    """
    if datetime_level is not None:
        print("WARNING: acqdatetime_series is deprecated. Use get_datetime instead")
    else:
        datetime_level = ["Acquisition1", "Acquisition", "Instance", "Series", "Study", "Content"]

    return get_datetime(dcmInfile, datetime_level)


def get_datetime(ds, datetime_level):
    """Return a datetime object from a pydicom Dataset.

    Arguments:
    ds - The pydicom Dataset object
    datetime_level - string or list of strings. The following values are valid:
                     'Instance', 'Study', 'Series', 'Acquisition', 'Acquisition1', 'Content'.

    If a single string is provided, a datetime object representing that level is
    returned if it exists in the DICOM header (ValueError is raised otherwise).
    If ``datetime_level`` is a list, the first string in the list which represents
    an existing DICOM tag in the Dataset is converted to datetime.
    """
    datetime_tags = {
        "Instance": ("0008,0012", "0008,0013"),  # Instance Creation D, Instance Creation T
        "Study": ("0008,0020", "0008,0030"),  # Study D, Study T
        "Series": ("0008,0021", "0008,0031"),  # Series D, Series T
        "Acquisition": ("0008,0022", "0008,0032"),  # Acq D, Acq T
        "Acquisition1": ("0008,002A",),  # Acq DT
        "Content": ("0008,0023", "0008,0033"),  # Content D, Content T
    }

    if not isinstance(datetime_level, (basestring, list, tuple)):
        raise ValueError(
            "get_datetime: datetime_level must be a string or list of strings, got {}".format(type(datetime_level)))

    if isinstance(datetime_level, basestring):
        datetime_level = [datetime_level]

    if len(datetime_level) == 0:
        raise ValueError("get_datetime: datetime_level not specified")

    for level in datetime_level:
        if level not in datetime_tags.keys():
            raise ValueError("get_datetime: datetime_level '{}' not in {}".format(level, list(datetime_tags.keys())))

    ### Extract the first valid DataElements
    data_elements = []
    for level in datetime_level:
        tags = datetime_tags[level]
        try:
            data_elements = [ds[dicom.tag.Tag(group_element_str.split(','))] for group_element_str in tags]
        except KeyError:
            "Tags not present in dataset, continue..."
            continue
        print("Using DateTime tag:", level)
        break

    if len(data_elements) == 0:
        raise ValueError("get_datetime: date/time tags for {} not present in DICOM file".format(datetime_level))

    print([(de.name, de.value) for de in data_elements])

    ### Concatenate date and time strings
    dtstring = data_elements[0].value
    if len(data_elements) == 2:
        dtstring += data_elements[1].value

    ### Parse the date-time string
    if '.' in dtstring and ('+' in dtstring or '-' in dtstring):
        dt = datetime.datetime.strptime(dtstring, '%Y%m%d%H%M%S.%f%z')  # YYYYMMDDHHMMSS.FFFFFF&ZZXX
    elif '+' in dtstring or '-' in dtstring:
        dt = datetime.datetime.strptime(dtstring, '%Y%m%d%H%M%S%z')  # YYYYMMDDHHMMSS&ZZXX
    elif '.' in dtstring:
        dt = datetime.datetime.strptime(dtstring, '%Y%m%d%H%M%S.%f')  # YYYYMMDDHHMMSS.FFFFFF
    else:
        dt = datetime.datetime.strptime(dtstring, '%Y%m%d%H%M%S')  # YYYYMMDDHHMMSS

    return dt
