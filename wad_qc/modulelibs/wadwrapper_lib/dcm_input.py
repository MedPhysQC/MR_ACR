"""
Functions to help input of dicom series.

This provides a number of DICOM handling routines
which are often used in plugins developed at UMCU.
This includes:
 * reading a 2D image
 * reading of a folder of 2D slices as a 3D image (excluding non-image files)
 * reading an Enhanced DICOM object as a 3D image
 * scaling data values according to slope and offset tags
 * accessing (nested) DICOM tags in 2D, 3D, and Enhanced DICOM

Changelog:
 20200508: split original dicomwrapper_lib.py v 20181104 into separate files;
"""

# US_RGB_USE_CHANNEL = 'R' # force using only the RED channel in RGB
# US_RGB_USE_CHANNEL = 'G' # force using only the GREEN channel in RGB
US_RGB_USE_CHANNEL = 'B'  # force using only the BLUE channel in RGB
# US_RGB_USE_CHANNEL = 'S' # force remove all data where there is a difference between R,G,B

try:
    import pydicom as dicom
    import pydicom.errors as dicomExceptions
except ImportError:
    import dicom

    try:  # moved to errors in 'bleeding edge' pydicom
        import dicom.errors as dicomExceptions
    except ImportError:
        import dicom.filereader as dicomExceptions

try:
    from . import pydicom_series as dcmseries
except ImportError:
    try:
        import pydicom_series as dcmseries
    except ImportError:
        from wad_qc.modulelibs import pydicom_series as dcmseries

import os
import numpy as np

stMode2D = "2D"
stMode3D = "3D"
stModeEnhanced = "Enhanced"
stModeBTO = "BTO"


### DICOM
# Helper functions
def _readSingleImage(filename, headers_only):
    """
    Internal function to read a single file as a DICOM image, optionally skipping pixel data.
    """

    dcmInfile = None
    try:
        dcmInfile = dicom.read_file(filename, stop_before_pixels=headers_only)  # Read only header!
    except dicomExceptions.InvalidDicomError:
        try:
            dcmInfile = dicom.read_file(filename, force=True, stop_before_pixels=headers_only)
        except:
            pass
    return dcmInfile


def _testIfEnhancedDICOM(filename):
    """
    Test if DICOM file to be read is an EnhancedDICOM object (but not BTO)
    """
    dcmInfile = _readSingleImage(filename, headers_only=True)
    if dcmInfile is None:
        # Not a dicom file
        return False

    try:
        sopclass = dcmInfile.SOPClassUID
    except AttributeError:
        # some other kind of dicom file and no reason to exclude it
        return False
    if sopclass in ['Enhanced MR Image Storage', '1.2.840.10008.5.1.4.1.1.4.1']:
        print('Image is Enhanced MR')
        return True
    elif sopclass in ['Enhanced CT Image Storage', '1.2.840.10008.5.1.4.1.1.2.1']:
        print('Image is Enhanced CT')
        return True
    else:
        if sopclass in ['Breast Tomosynthesis Image Storage', '1.2.840.10008.5.1.4.1.1.13.1.3']:
            print('Image is BTO')
        return False


def _readGroupSequenceTag(sequencetags, keytag):
    """
    Return value of tag in a sequence of tags of a EnhancedDICOM header object.
    Sequence should be either a SharedFunctionalGroupsSequence or a PerFrameFunctionalGroupsSequence
    """
    value = ''
    for seq in sequencetags.keys():
        subsetA = sequencetags[(seq.group, seq.elem)]
        if subsetA.VR != 'SQ' or subsetA.VM == 0:
            continue
        subsetA = sequencetags[(seq.group, seq.elem)][0]
        for elemA in subsetA.keys():
            if (elemA.group, elemA.elem) == (keytag.group, keytag.elem):
                value = subsetA[(elemA.group, elemA.elem)].value
    return value


def _removeBogusDICOMfiles(instancedict):
    """
    When a folder of files is offered, prepare a list of files that are valid DICOM image files;
    this removes some report files that would break reading a folder of slices as a 3D image
    """
    results = []
    for fname in instancedict:
        skipMe = False
        dcmInfile = _readSingleImage(fname, headers_only=True)
        if dcmInfile is None:
            # Not a dicom file
            continue

        # AS: Skip the file if it is of SOPClassUID 1.2.840.10008.5.1.4.1.1.66
        try:
            sopclass = dcmInfile.SOPClassUID
        except AttributeError:
            # some other kind of dicom file and no reason to exclude it
            results.append(fname)
            continue
        if sopclass in ['Raw Data Storage', '1.2.840.10008.5.1.4.1.1.66']:
            print('Skipping RAW file %s' % fname)
            continue
        else:
            results.append(fname)

    return results


def getDICOMMode(dcmInfile):
    """
    Determine the structure of the DICOM image: 2D, 3D, Enhanced, BTO
    """
    dcmMode = stMode3D
    try:
        num = len(dcmInfile._datasets)
    except AttributeError:
        dcmMode = stMode2D
        try:
            sopclass = dcmInfile.SOPClassUID
            if sopclass in ['Enhanced MR Image Storage', '1.2.840.10008.5.1.4.1.1.4.1']:
                dcmMode = stModeEnhanced
            elif sopclass in ['Enhanced CT Image Storage', '1.2.840.10008.5.1.4.1.1.2.1']:
                dcmMode = stModeEnhanced
            elif sopclass in ['Breast Tomosynthesis Image Storage', '1.2.840.10008.5.1.4.1.1.13.1.3']:
                dcmMode = stModeBTO

        except AttributeError:
            pass
    return dcmMode


def readDICOMtag(key, dcmInfile, imslice=0):  # slice=2 is image 3
    """
    Returns the value of the dicomtag 'key' for image slice 'imslice'
    imslice is ignored for 2D images.
    A key is defined as e.g. "0018,11A0", but could also be a nested tag "0018,11A0,0008,1001"
    Returns empty string if tag is not defined in the dicomfile.
    """

    # transcribe "0018,11A0" as a tag entry
    keytag = dicom.tag.Tag(key.split(',')[0], key.split(',')[1])

    if len(key.split(',')) > 2:  # key has more than 2 groups of digits, so presumable a nested tag
        return readNestedDICOMtag(key, dcmInfile, imslice)
    try:
        dicomMode = getDICOMMode(dcmInfile)
        if dicomMode == stMode2D:
            value = dcmInfile[keytag].value
        elif dicomMode == stMode3D:
            value = dcmInfile._datasets[imslice][keytag].value
        else:  # EnhancedDicom or BTO
            try:
                value = dcmInfile[keytag].value
            except:
                value = ''

            if value == '':  # not found, look for it in sharedtags
                value = _readGroupSequenceTag(dcmInfile.SharedFunctionalGroupsSequence[0], keytag)

            if value == '':  # not found, look for it in perframetags
                value = _readGroupSequenceTag(dcmInfile.PerFrameFunctionalGroupsSequence[imslice], keytag)

            if value == '':  # not found, look for it in perframetags
                print("[ww_readDICOMtag] key", key, "not found")

    except:
        print("[ww_readDICOMtag] Exception for key", key)
        value = ""
    return value


def readNestedDICOMtag(key, dcmInfile, imslice):  # slice=2 is image 3
    """
    Returns the value of the nested dicomtag 'key' for image slice 'imslice'
    imslice is ignored for 2D images.
    A nested tag is defined as e.g "0018,11A0,0008,1001" meaning tag "0008,1001" in subset tag "0018,11A0"
    Returns empty string if tag is not defined in the dicomfile.
    """
    # transcribe "0018,11A0" as a tag entry
    lv0 = dicom.tag.Tag(key.split(',')[0], key.split(',')[1])
    lv1 = dicom.tag.Tag(key.split(',')[2], key.split(',')[3])

    dicomMode = getDICOMMode(dcmInfile)
    try:
        if dicomMode == stMode2D:
            value = dcmInfile[lv0][0][lv1].value
        elif dicomMode == stMode3D:
            value = dcmInfile._datasets[imslice][lv0][0][lv1].value
        else:  # EnhancedDicom or BTO
            # sharedTags = dcmInfile.SharedFunctionalGroupsSequence[0]
            try:
                value = dcmInfile[lv0][0][lv1].value
            except:
                value = ''
            if value == '':
                sharedTags = dcmInfile.SharedFunctionalGroupsSequence[0]
                for seq in sharedTags.dir():
                    subsetA = sharedTags.data_element(seq)[0]
                    for elemA in subsetA.keys():
                        if (elemA.group, elemA.elem) == (lv0.group, lv0.elem):
                            subsetB = subsetA[(elemA.group, elemA.elem)]
                            if subsetB.VM > 0 and subsetB.VR == 'SQ':
                                for elemB in subsetB[0].keys():
                                    if (elemB.group, elemB.elem) == (lv1.group, lv1.elem):
                                        value = elemB.value


    except:
        print("[ww_readNestedDICOMtag] Exception for key", key, dicomMode)
        value = ""
    return value


def prepareEnhancedInput(filename, headers_only, do_transpose=True, logTag="[prepareEnhancedInput] "):
    """
    Reads filename as an EnhancedDICOM object. If not headers_only , scaling the pixel values according to RescaleIntercept and RescaleIntercept of each frame
    and transposing for pyqtgraph format.
    Raises ValueError if file cannot be opened as a DICOM object
    Returns raw dcmfile, scaled and transposed pixeldata (or None), type of DICOM object.
    """
    dcmInfile = None
    pixeldataIn = None

    # Repair Mapping for MR; not correct for all scanner
    keymapping = [
        ("0040,9096,0040,9224", "0028,1052"),  # Real World Value Intercept -> Rescale Intercept
        ("0040,9096,0040,9225", "0028,1053"),  # Real World Value Slope -> Rescale Slope
    ]

    dcmInfile = _readSingleImage(filename, headers_only=headers_only)
    if dcmInfile is None:
        # Not a dicom file
        raise ValueError("{} ERROR! {} is not a valid Enhanced DICOM object".format(logTag, filename))

    if not headers_only:
        pixeldataIn = dcmInfile.pixel_array.astype(int)
        perframe = dcmInfile.PerFrameFunctionalGroupsSequence  # Sequence of perframeTags
        for i in range(len(perframe)):
            intercept = perframe[i].PixelValueTransformationSequence[0].RescaleIntercept
            slope = perframe[i].PixelValueTransformationSequence[0].RescaleSlope
            pixeldataIn[i] = intercept + slope * pixeldataIn[i]
        if do_transpose:
            pixeldataIn = np.transpose(pixeldataIn, (0, 2, 1))

    return dcmInfile, pixeldataIn, getDICOMMode(dcmInfile)


def prepareInput(instancedict, headers_only, do_transpose=True, logTag="[prepareInput] ", skip_check=False,
                 splitOnPosition=True, rgbchannel=US_RGB_USE_CHANNEL):
    """
    Reads inputfile as an EnhancedDICOM object; if an EnhancedDICOM object is detected, prepareEnhancedInput is called.
    Checks if the input is as expected: number of slices etc.
    If not headers_only, scaling the pixel values according to RescaleIntercept and RescaleIntercept of each frame
    and transposing for pyqtgraph format.
    Raises ValueError if file cannot be opened as a DICOM object
    Returns raw dcmfile, scaled and transposed pixeldata (or None), type of DICOM object.

    Speed up option skip_check added: no checking for bogus files or multi path
    Option to prevent default splitting of series on patientposition differences
    """
    # compile a list of valid files to read
    if not skip_check:
        instancedict = _removeBogusDICOMfiles(instancedict)

    dcmInfile = None
    pixeldataIn = None

    # Repair Mapping for MR; not correct for all scanner
    keymapping = [
        ("0040,9096,0040,9224", "0028,1052"),  # Real World Value Intercept -> Rescale Intercept
        ("0040,9096,0040,9225", "0028,1053"),  # Real World Value Slope -> Rescale Slope
    ]

    # Check if input data is as expected: MR needs 3D data
    if len(instancedict) == 1:
        ModeEnhanced = False
        ModeEnhanced = _testIfEnhancedDICOM(instancedict[0])
        if ModeEnhanced:
            return prepareEnhancedInput(instancedict[0], do_transpose=do_transpose,
                                        headers_only=headers_only)  # scaled and transposed

        if dcmInfile is None:
            filename = instancedict[0]
            dcmInfile = _readSingleImage(filename, headers_only=headers_only)
            if dcmInfile is None:
                # Not a dicom file
                raise ValueError("{} ERROR! {} is not a valid non-Enhanced DICOM object".format(logTag, filename))

            modality = dcmInfile.Modality

            if not headers_only:
                # need scaling for single slice, already done for series in pydicom_series, but not for some MR files
                if modality == 'CT' or modality == 'MR':
                    if not "RescaleIntercept" in dcmInfile:  # in wrong place define for some MR files
                        dcmInfile.RescaleIntercept = readDICOMtag(keymapping[0][0], dcmInfile, 0)
                        if dcmInfile.RescaleIntercept == "":
                            dcmInfile.RescaleIntercept = 0
                        dcmInfile.RescaleSlope = readDICOMtag(keymapping[1][0], dcmInfile, 0)
                        if dcmInfile.RescaleSlope == "":
                            dcmInfile.RescaleSlope = 1

                    pixeldataIn = np.int16(dcmInfile.pixel_array.astype(int))
                    slope = dcmInfile.RescaleSlope
                    intercept = dcmInfile.RescaleIntercept
                    if int(slope) != slope or int(intercept) != intercept:
                        pixeldataIn = pixeldataIn.astype(float)
                    pixeldataIn = intercept + slope * pixeldataIn
                elif modality == 'MG' and getDICOMMode(dcmInfile) == stModeBTO:
                    print('!WARNING! MG BTO dataset! DICOM info is NOT properly adjusted, no scaling applied yet!')
                    pixeldataIn = dcmInfile.pixel_array
                elif modality == 'MG' or modality == 'CR' or modality == 'DX':
                    pixeldataIn = dcmInfile.pixel_array
                elif modality == 'RF' or modality == 'XA':  # fixme! 2D and 3D
                    pixeldataIn = dcmInfile.pixel_array
                elif modality == 'US':
                    rgbmode = (dcmInfile.SamplesPerPixel == 3)
                    if not rgbmode:
                        if len(np.shape(dcmInfile.pixel_array)) == 2:
                            pixeldataIn = dcmInfile.pixel_array
                        elif len(np.shape(dcmInfile.pixel_array)) == 3:
                            pixeldataIn = dcmInfile.pixel_array
                            pixeldataIn = pixeldataIn[0]
                    else:
                        # AS: this fix was only needed in pydicom < 1.0; solved in later versions
                        try:
                            dicomversion = int(dicom.__version_info__[0])
                        except:
                            dicomversion = 0
                        if dicomversion == 0:
                            try:
                                nofframes = dcmInfile.NumberOfFrames
                            except AttributeError:
                                nofframes = 1
                            if dcmInfile.PlanarConfiguration == 0:
                                pixel_array = dcmInfile.pixel_array.reshape(nofframes, dcmInfile.Rows,
                                                                            dcmInfile.Columns,
                                                                            dcmInfile.SamplesPerPixel)
                            else:
                                pixel_array = dcmInfile.pixel_array.reshape(dcmInfile.SamplesPerPixel, nofframes,
                                                                            dcmInfile.Rows, dcmInfile.Columns)
                        else:
                            pixel_array = dcmInfile.pixel_array

                        # pick channel to use for RGB
                        print('Using RGB Channel {}'.format(rgbchannel))
                        if rgbchannel == 'S':  # remove all data where there is a difference between R,G,B
                            if len(np.shape(pixel_array)) == 3:  # 2d rgb
                                pixeldataIn = pixel_array[:, :, 0]
                                pixeldataInR = pixel_array[:, :, 0]
                                pixeldataInG = pixel_array[:, :, 1]
                                pixeldataInB = pixel_array[:, :, 2]
                            elif len(np.shape(pixel_array)) == 4:  # 3d rgb
                                pixeldataIn = pixel_array[-1, :, :, 0]
                                pixeldataInR = pixel_array[-1, :, :, 0]
                                pixeldataInG = pixel_array[-1, :, :, 1]
                                pixeldataInB = pixel_array[-1, :, :, 2]

                            # remove rgb info
                            for y in range(dcmInfile.Rows):
                                for x in range(dcmInfile.Columns):
                                    r = pixeldataInR[y, x]
                                    g = pixeldataInG[y, x]
                                    b = pixeldataInB[y, x]
                                    ma = max(r, g, b)
                                    mi = min(r, g, b)
                                    if ma != mi:
                                        pixeldataIn[y, x] = 0
                        else:
                            if rgbchannel == 'R':
                                chan = 0
                            elif rgbchannel == 'G':
                                chan = 1
                            elif rgbchannel == 'B':
                                chan = 2
                            if len(np.shape(pixel_array)) == 3:  # 2d rgb
                                pixeldataIn = pixel_array[:, :, chan]
                            elif len(np.shape(pixel_array)) == 4:  # 3d rgb
                                if dcmInfile.PlanarConfiguration == 0:
                                    pixeldataIn = pixel_array[-1, :, :, chan]
                                else:
                                    pixeldataIn = pixel_array[chan, -1, :, :]  # adjusted for ALOKA images
                else:
                    raise ValueError("ERROR! reading of modality {} not implemented!".format(modality))

    else:
        if not skip_check:
            path = os.path.dirname(instancedict[0])
            for ip in instancedict:
                if os.path.dirname(ip) != path:
                    raise ValueError(
                        "{} ERROR! multiple would-be dicom files scattered over multiple dirs!".format(logTag))

        try:
            dcmInfile = dcmseries.read_files(instancedict, True, readPixelData=False, splitOnPosition=splitOnPosition)[
                0]
            if not headers_only:  # NOTE: Rescaling is already done pydicom_series, but maybe not for stupid MR
                modality = dcmInfile._datasets[0].Modality
                if modality == 'CT' or modality == 'MR':
                    for i in range(len(dcmInfile._datasets)):
                        if not "RescaleIntercept" in dcmInfile._datasets[i]:  # in wrong place define for some MR files
                            dcmInfile._datasets[i].RescaleIntercept = readDICOMtag(keymapping[0][0], dcmInfile, i)
                            dcmInfile._datasets[i].RescaleSlope = readDICOMtag(keymapping[1][0], dcmInfile, i)
                pixeldataIn = dcmInfile.get_pixel_array()

        except Exception as e:
            raise ValueError("{} ERROR! {} is not a valid non-Enhanced DICOM series ({})".format(logTag,
                                                                                                 os.path.dirname(
                                                                                                     instancedict[0]),
                                                                                                 str(e)))

    if not headers_only and do_transpose:
        ndims = len(np.shape(pixeldataIn))
        if ndims == 3:
            pixeldataIn = np.transpose(pixeldataIn, (0, 2, 1))
        elif ndims == 2:
            pixeldataIn = pixeldataIn.transpose()

    return dcmInfile, pixeldataIn, getDICOMMode(dcmInfile)
