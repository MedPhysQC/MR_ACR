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


#--------------------------------------
# Modified from skimage.measure.fit.py
# Copyright: 2009-2022 the scikit-image team
# Licensed under BSD-3-Clause
#
# Modifications AmsUMC - JK
# - do not rotate ellipse, remove x*y term
# - always return ellipse phi=0, i.e. a along x-axis and b along y-axis
#--------------------------------------

import math
import numpy as np

from numpy.linalg import inv
from scipy import optimize

def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError(f"Input data must have shape (N, {dim}).")


def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError('Input data must be at least 2D.')


class BaseModel:
    def __init__(self):
        self.params = None


class MyEllipseModel(BaseModel):
    """Total least squares estimator for 2D ellipses.

    The functional model of the ellipse is::

        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)

    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.

    The estimator is based on a least squares minimization. The optimal
    solution is computed directly, no iterations are required. This leads
    to a simple, stable and robust fitting method.

    The ``params`` attribute contains the parameters in the following order::

        xc, yc, a, b, theta

    Attributes
    ----------
    params : tuple
        Ellipse model parameters in the following order `xc`, `yc`, `a`, `b`,
        `theta`.

    Examples
    --------

    >>> xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25),
    ...                                params=(10, 15, 8, 4, np.deg2rad(30)))
    >>> ellipse = EllipseModel()
    >>> ellipse.estimate(xy)
    True
    >>> np.round(ellipse.params, 2)
    array([10.  , 15.  ,  8.  ,  4.  ,  0.52])
    >>> np.round(abs(ellipse.residuals(xy)), 5)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def estimate(self, data):
        """Estimate ellipse model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.


        References
        ----------
        .. [1] Halir, R.; Flusser, J. "Numerically stable direct least squares
               fitting of ellipses". In Proc. 6th International Conference in
               Central Europe on Computer Graphics and Visualization.
               WSCG (Vol. 98, pp. 125-132).

        """
        # Original Implementation: Ben Hammel, Nick Sullivan-Molina
        # another REFERENCE: [2] http://mathworld.wolfram.com/Ellipse.html
        _check_data_dim(data, dim=2)

        # to prevent integer overflow, cast data to float, if it isn't already
        float_type = np.promote_types(data.dtype, np.float32)
        data = data.astype(float_type, copy=False)

        # normalize value range to avoid misfitting due to numeric errors if
        # the relative distanceses are small compared to absolute distances
        origin = data.mean(axis=0)
        data = data - origin
        scale = data.std()
        if scale < np.finfo(float_type).tiny:
            print(
                "Standard deviation of data is too small to estimate "
                "ellipse with meaningful precision."
            )
            return False
        data /= scale

        x = data[:, 0]
        y = data[:, 1]

        # Quadratic part of design matrix [eqn. 15] from [1]
        # AmsUMC - JK - do not rotate ellipse, remove x*y term
            #D1 = np.vstack([x**2, x * y, y**2]).T
        D1 = np.vstack([x**2, y**2]).T
        # Linear part of design matrix [eqn. 16] from [1]
        D2 = np.vstack([x, y, np.ones_like(x)]).T

        # forming scatter matrix [eqn. 17] from [1]
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2

        # Constraint matrix [eqn. 18]
        # AmsUMC - JK - do not rotate ellipse, remove x*y term
            #C1 = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
        C1 = np.array([[0.0, 2.0], [2.0, 0.0]])

        try:
            # Reduced scatter matrix [eqn. 29]
            M = inv(C1) @ (S1 - S2 @ inv(S3) @ S2.T)
        except np.linalg.LinAlgError:  # LinAlgError: Singular matrix
            return False

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors
        # from this equation [eqn. 28]
        eig_vals, eig_vecs = np.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        # AmsUMC - JK - do not rotate ellipse, remove x*y term
            # cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[2, :]) - np.power(
            #     eig_vecs[1, :], 2
            # )
        cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[1, :])
        a1 = eig_vecs[:, (cond > 0)]
        # seeks for empty matrix
        # AmsUMC - JK - do not rotate ellipse, remove x*y term
            #if 0 in a1.shape or len(a1.ravel()) != 3:
        if 0 in a1.shape or len(a1.ravel()) != 2:
            return False
        # AmsUMC - JK - do not rotate ellipse, remove x*y term
            #a, b, c = a1.ravel()
        a, c = a1.ravel()
        b = 0

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -inv(S3) @ S2.T @ a1
        d, f, g = a2.ravel()

        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 (eqn. 15) from [2]
        b /= 2.0
        d /= 2.0
        f /= 2.0

        # finding center of ellipse [eqn.19 and 20] from [2]
        x0 = (c * d - b * f) / (b**2.0 - a * c)
        y0 = (a * f - b * d) / (b**2.0 - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from [2]
        numerator = a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g
        term = np.sqrt((a - c) ** 2 + 4 * b**2)
        denominator1 = (b**2 - a * c) * (term - (a + c))
        denominator2 = (b**2 - a * c) * (-term - (a + c))
        width = np.sqrt(2 * numerator / denominator1)
        height = np.sqrt(2 * numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse
        # to x-axis [eqn. 23] from [2].
        phi = 0.5 * np.arctan((2.0 * b) / (a - c))
        if a > c:
            phi += 0.5 * np.pi

        # stabilize parameters:
        # sometimes small fluctuations in data can cause
        # height and width to swap
        if width < height:
            width, height = height, width
            phi += np.pi / 2

        phi %= np.pi

        # AmsUMC - JK - do not rotate ellipse, remove x*y term
        # phi should be either 0 or pi/2 now
        # force phi=0 by swapping the long and short axis for pi/2
        if phi > np.pi / 4:
            width, height = height, width
            phi -= np.pi / 2

        # revert normalization and set params
        params = np.nan_to_num([x0, y0, width, height, phi]).real
        params[:4] *= scale
        params[:2] += origin

        self.params = tuple(float(p) for p in params)

        return True

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the ellipse is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N,) array
            Residual for each data point.

        """

        _check_data_dim(data, dim=2)

        xc, yc, a, b, theta = self.params

        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        def fun(t, xi, yi):
            ct = math.cos(np.squeeze(t))
            st = math.sin(np.squeeze(t))
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt) ** 2 + (yi - yt) ** 2

        # def Dfun(t, xi, yi):
        #     ct = math.cos(t)
        #     st = math.sin(t)
        #     xt = xc + a * ctheta * ct - b * stheta * st
        #     yt = yc + a * stheta * ct + b * ctheta * st
        #     dfx_t = - 2 * (xi - xt) * (- a * ctheta * st
        #                                - b * stheta * ct)
        #     dfy_t = - 2 * (yi - yt) * (- a * stheta * st
        #                                + b * ctheta * ct)
        #     return [dfx_t + dfy_t]

        residuals = np.empty((N,), dtype=np.float64)

        # initial guess for parameter t of closest point on ellipse
        t0 = np.arctan2(y - yc, x - xc) - theta

        # determine shortest distance to ellipse for each point
        for i in range(N):
            xi = x[i]
            yi = y[i]
            # faster without Dfun, because of the python overhead
            t, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
            residuals[i] = np.sqrt(fun(t, xi, yi))

        return residuals

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5,) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """

        if params is None:
            params = self.params

        xc, yc, a, b, theta = params

        ct = np.cos(t)
        st = np.sin(t)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)
