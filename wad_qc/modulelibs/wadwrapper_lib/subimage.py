"""
Function for extracting parts of images, without having to think about boundaries.

Changelog:
  20200508: split from wadwrapper_lib.py
"""
import numpy as np


def extract(Z, shape, position, fill=np.NaN):
    """ Extract a sub-array from Z using given shape and centered on position.
      If some part of the sub-array is out of Z bounds, result is padded
         with fill value.

         **Parameters**
             `Z` : array_like
                Input array.

            `shape` : tuple
                Shape of the output array

            `position` : tuple
                Position within Z

            `fill` : scalar
                Fill value

         **Returns**
             `out` : array_like
                 Z slice with given shape and center

         **Examples**

         >>> Z = np.arange(0,16).reshape((4,4))
         >>> extract(Z, shape=(3,3), position=(0,0))
         [[ NaN  NaN  NaN]
          [ NaN   0.   1.]
          [ NaN   4.   5.]]

         Schema::


              +-----------+
              | 0   0   0 |  =  extract (Z, shape=(3,3), position=(0,0))
              |   +-------|---------+
              | 0 | 0   1 | 2   3   |  =  Z
              |   |       |         |
              | 0 | 4   5 | 6   7   |
              +-----------+         |
                  | 8    9   10  11 |
                  |                 |
                  | 12   13  14  15 |
                  +-----------------+


         >>> Z = np.arange(0,16).reshape((4,4))
         >>> extract(Z, shape=(3,3), position=(3,3))
         [[ 10.  11.  NaN]
          [ 14.  15.  NaN]
          [ NaN  NaN  NaN]]

         Schema::

             +---------------+
             | 0   1   2   3 | = Z
             |               |
             | 4   5   6   7 |
             |       +-----------+
             | 8   9 |10  11 | 0 | = extract (Z, shape=(3,3),position=(3,3))
             |       |       |   |
             | 12 13 |14  15 | 0 |
             +---------------+   |
                     | 0   0   0 |
                     +-----------+
    """
    #    assert(len(position) == len(Z.shape))
    #    if len(shape) < len(Z.shape):
    #        shape = shape + Z.shape[len(Z.shape)-len(shape):]

    R = np.ones(shape, dtype=Z.dtype) * fill
    P = np.array(list(position)).astype(int)
    Rs = np.array(list(R.shape)).astype(int)
    Zs = np.array(list(Z.shape)).astype(int)

    R_start = np.zeros((len(shape),)).astype(int)
    R_stop = np.array(list(shape)).astype(int)
    Z_start = (P - Rs // 2)
    Z_stop = (P + Rs // 2) + Rs % 2

    R_start = (R_start - np.minimum(Z_start, 0)).tolist()
    Z_start = (np.maximum(Z_start, 0)).tolist()
    R_stop = (R_stop - np.maximum(Z_stop - Zs, 0)).tolist()
    Z_stop = (np.minimum(Z_stop, Zs)).tolist()

    r = [slice(start, stop) for start, stop in zip(R_start, R_stop)]
    z = [slice(start, stop) for start, stop in zip(Z_start, Z_stop)]

    R[tuple(r)] = Z[tuple(z)]

    return R
