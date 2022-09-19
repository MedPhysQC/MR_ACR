"""
Functions for image thresholding.

Changelog:
  20200508: split from wadwrapper_lib.py
"""
import numpy as np
import scipy.ndimage as scind


### thresholding
def threshold_adaptive(image, block_size, method='gaussian', offset=0,
                       mode='reflect', param=None):
    """
    from skitimage 0.8
    Applies an adaptive threshold to an array.

    Also known as local or dynamic thresholding where the threshold value is
    the weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a a
    given function using the 'generic' method.

    Parameters
    ==========
    image : (N, M) ndarray
        Input image.
    block_size : int
        Uneven size of pixel neighborhood which is used to calculate the
        threshold value (e.g. 3, 5, 7, ..., 21, ...).
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighbourhood in
        weighted mean image.

        * 'generic': use custom function (see `param` parameter)
        * 'gaussian': apply gaussian filter (see `param` parameter for custom\
                      sigma value)
        * 'mean': apply arithmetic mean filter
        * 'median': apply median rank filter

        By default the 'gaussian' method is used.
    offset : float, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'.
    param : {int, function}, optional
        Either specify sigma for 'gaussian' method or function object for
        'generic' method. This functions takes the flat array of local
        neighbourhood as a single argument and returns the calculated
        threshold for the centre pixel.

    Returns
    =======
    threshold : (N, M) ndarray
        Thresholded binary image

    References
    ==========
    .. [1] http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#adaptivethreshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> binary_image1 = threshold_adaptive(image, 15, 'mean')
    >>> func = lambda arr: arr.mean()
    >>> binary_image2 = threshold_adaptive(image, 15, 'generic', param=func)
    """
    thresh_image = np.zeros(image.shape, 'double')
    if method == 'generic':
        scind.generic_filter(image, param, block_size,
                             output=thresh_image, mode=mode)
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = (block_size - 1) / 6.0
        else:
            sigma = param
        scind.gaussian_filter(image, sigma, output=thresh_image,
                              mode=mode)
    elif method == 'mean':
        mask = 1. / block_size * np.ones((block_size,))
        # separation of filters to speedup convolution
        scind.convolve1d(image, mask, axis=0, output=thresh_image,
                         mode=mode)
        scind.convolve1d(thresh_image, mask, axis=1,
                         output=thresh_image, mode=mode)
    elif method == 'median':
        scind.median_filter(image, block_size, output=thresh_image,
                            mode=mode)

    return image > (thresh_image - offset)


def __IJIsoData(data):
    """
    This is the original ImageJ IsoData implementation
    """
    count0 = data[0]
    data[0] = 0  # set to zero so erased areas aren't included
    countMax = data[-1]
    data[-1] = 0

    maxValue = len(data) - 1
    amin = 0
    while data[amin] == 0 and amin < maxValue:
        amin += 1
    amax = maxValue
    while data[amax] == 0 and amax > 0:
        amax -= 1
    if amin >= amax:
        data[0] = count0
        data[maxValue] = countMax;
        level = len(data) / 2
        return level

    movingIndex = amin
    cond = True
    while cond:
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        sum4 = 0.0
        for i in range(amin, movingIndex + 1):
            sum1 += i * data[i]
            sum2 += data[i]

        for i in range(movingIndex + 1, amax + 1):
            sum3 += i * data[i]
            sum4 += data[i]

        result = (sum1 / sum2 + sum3 / sum4) / 2.0
        movingIndex += 1
        cond = ((movingIndex + 1) <= result and movingIndex < amax - 1)

    data[0] = count0
    data[maxValue] = countMax;
    level = int(round(result))
    return level


def threshold_isodata2(data):
    """
    Ripped from ImageJ "defaultIsoData"
    """
    maxCount = np.max(data)
    mode = np.argmax(data)

    data2 = np.copy(data)
    maxCount2 = 0
    for i, v in enumerate(data2):
        if v > maxCount2 and i != mode:
            maxCount2 = v

    if maxCount > maxCount2 * 2 and maxCount2 != 0:
        data2[mode] = int(maxCount2 * 1.5)

    return __IJIsoData(data2)
