"""
Rotation/Translation operations for data in the spherical harmonic domains
"""

import numpy as np
import spharpy
from scipy.special import eval_jacobi, factorial
from scipy.spatial.transform import Rotation


def rotation_z_axis(n_max, angle):
    """Rotation matrix for complex spherical harmonics around the z-axis
    by a given angle. The rotation is performed such that positive angles
    result in a counter clockwise rotation of the data [1]_.

    .. math::

        c_{nm}(\\theta, \\phi + \\xi) = e^{-im\\xi} c_{nm}(\\theta, \\phi)

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    angle : number
        Rotation angle in radians `[0, 2 \\pi]`

    Returns
    -------
    array, complex
        Diagonal rotation matrix evaluated for the specified angle

    References
    ----------
    .. [1]  N. A. Gumerov and R. Duraiswami, “Recursions for the computation
            of multipole translation and rotation coefficients for the 3-d
            helmholtz equation,” vol. 25, no. 4, pp. 1344–1381, 2003.


    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 1, 0, 0])
    >>> rotMat = spharpy.transforms.rotation_z_axis(n_max, np.pi/2)
    >>> sh_vec_rotated = rotMat @ sh_vec
    """

    acn = np.arange(0, (n_max+1)**2)
    m = spharpy.spherical.acn2nm(acn)[1]
    rotation_phi = np.exp(-1j*angle*m)

    return np.diag(rotation_phi)


def rotation_z_axis_real(n_max, angle):
    """Rotation matrix for real-valued spherical harmonics around the z-axis
    by a given angle. The rotation is performed such that positive angles
    result in a counter clockwise rotation of the data [1]_.

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    angle : number
        Rotation angle in radians `[0, 2 \\pi]`

    Returns
    -------
    rotation_matrix : ndarray
        Block-diagonal Rotation matrix evaluated for the specified angle.

    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 1, 0, 0])
    >>> rotMat = spharpy.transforms.rotation_z_axis_real(n_max, np.pi/2)
    >>> sh_vec_rotated = rotMat @ sh_vec

    """
    acn = np.arange(0, (n_max + 1) ** 2)
    n, m = spharpy.spherical.acn2nm(acn)
    acn_reverse_degree = n ** 2 + n - m

    rotation_phi = np.zeros(((n_max + 1) ** 2, (n_max + 1) ** 2))
    mask = m == 0
    rotation_phi[acn[mask], acn[mask]] = 1.0

    mask_pos = m > 0
    mask_neg = m < 0
    rotation_phi[acn[mask_pos], acn[mask_pos]] = np.cos(
        np.abs(m[mask_pos]) * angle)
    rotation_phi[acn[mask_neg], acn[mask_neg]] = np.cos(
        np.abs(m[mask_neg]) * angle)

    # non diagonal
    rotation_phi[acn[mask_pos], acn_reverse_degree[mask_pos]] = -np.sin(
        np.abs(m[mask_pos]) * angle)
    rotation_phi[acn[mask_neg], acn_reverse_degree[mask_neg]] = np.sin(
        np.abs(m[mask_neg]) * angle)

    return rotation_phi


def wigner_d_rotation(n_max, alpha, beta, gamma):
    r"""Wigner-D rotation matrix for Euler rotations by angles
    (\alpha, \beta, \gamma) around the (z,y,z)-axes. The rotation is
    performed such that positive angles result in a counter clockwise rotation
    of the data.

    .. math::

        D_{m^\dash,m}^n(\alpha, \beta, \gamma) =
        e^{-im^\dash\alpha} d_{m^\dash,m}^n(\beta) e^{-im\gamma}

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    alpha : float
        First z-axis rotation angle
    beta : float
        Y-axis rotation angle
    gamma : float
        Second z-axis rotation angle

    Returns
    -------
    array, complex
        Block diagonal rotation matrix


    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 0, 1, 0])
    >>> rotMat = spharpy.transforms.wigner_d_rotation(n_max, 0, np.pi/4, 0)
    >>> sh_vec_rotated = rotMat @ sh_vec

    References
    ----------
    .. [#] B. Rafaely, Fundamentals of Spherical Array Processing, 1st ed.,
           vol. 8. Springer-Verlag GmbH Berlin Heidelberg, 2015.

    """

    n_sh = (n_max+1)**2

    R = np.zeros((n_sh, n_sh), dtype=complex)

    for row in np.arange(0, (n_max+1)**2):
        n_dash, m_dash = spharpy.spherical.acn2nm(row)
        for column in np.arange(0, (n_max+1)**2):
            n, m = spharpy.spherical.acn2nm(column)
            if n == n_dash:
                rot_alpha = np.exp(-1j*m_dash*alpha)
                rot_beta = wigner_d_function(n, m_dash, m, beta)
                rot_gamma = np.exp(-1j*m*gamma)
                R[row, column] = rot_alpha * rot_beta * rot_gamma

    return R


def wigner_d_rotation_real(n_max, alpha, beta, gamma):
    r"""Wigner-D rotation matrix for Euler rotations for real-valued spherical
    harmonics by angles (\alpha, \beta, \gamma) around the (z,y,z)-axes.
    The rotation is performed such that positive angles result in a counter
    clockwise rotation of the data.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    alpha : float
        First z-axis rotation angle
    beta : float
        Y-axis rotation angle
    gamma : float
        Second z-axis rotation angle

    Returns
    -------
    array, float
        Block diagonal rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 0, 1, 0])
    >>> rotMat = spharpy.transforms.wigner_d_rotation_real(
    >>>     n_max, 0, np.pi/4, 0)
    >>> sh_vec_rotated = rotMat @ sh_vec

    References
    ----------
    .. [#]  M. A. Blanco, M. Flórez, and M. Bermejo, “Evaluation of the
            rotation matrices in the basis of real spherical harmonics,”
            Journal of Molecular Structure: THEOCHEM,  vol. 419, no. 1–3,
            pp. 19–27, Dec. 1997, doi: 10.1016/S0166-1280(97)00185-1.

    """
    n_sh = (n_max+1)**2
    R = np.zeros((n_sh, n_sh), dtype=np.double)

    for row_acn in np.arange(0, (n_max+1)**2):
        for col_acn in np.arange(0, (n_max+1)**2):
            n, m = spharpy.spherical.acn2nm(col_acn)
            n_dash, m_dash = spharpy.spherical.acn2nm(row_acn)
            if n == n_dash:
                # minus beta opposite rotation direction
                d_l_1 = wigner_d_function(n, np.abs(m_dash), np.abs(m), -beta)
                d_l_2 = wigner_d_function(n, np.abs(m), -np.abs(m_dash), -beta)

                R[row_acn, col_acn] = \
                    _sign(m_dash) * _Phi(m, alpha) * _Phi(m_dash, gamma) * \
                    (d_l_1 + (-1)**int(m) * d_l_2)/2 \
                    - _sign(m) * _Phi(-m, alpha) * _Phi(-m_dash, gamma) * \
                    (d_l_1 - (-1)**int(m) * d_l_2)/2

    return R


def _sign(x):
    """
    Returns sign of x, differs from numpy definition for x=0
    """
    if x < 0:
        sign = -1
    else:
        sign = 1

    return sign


def _Phi(m, angle):
    """
    Rotation Matrix around z-axis for real Spherical Harmonics as defined in
    Blanco et al., Evaluation of the rotation matrices in the basis of real
    spherical harmonics, eq.(8)
    """
    if m > 0:
        phi = np.sqrt(2)*np.cos(m*angle)
    elif m == 0:
        phi = 1
    elif m < 0:
        # minus due to differing phase convention
        phi = -np.sqrt(2)*np.sin(np.abs(m)*angle)

    return phi


def wigner_d_function(n, m_dash, m, beta):
    r"""Wigner-d function for rotations around the y-axis as defined in [#]_.

    Parameters
    ----------
    n : int
        order
    m_dash : int
        degree
    m : int
        degree
    beta : float
        Rotation angle

    Returns
    -------
    float
        Wigner-d symbol

    References
    ----------
    .. [#] B. Rafaely, Fundamentals of Spherical Array Processing, 1st ed.,
           vol. 8. Springer-Verlag GmbH Berlin Heidelberg, 2015.

    """

    if m >= m_dash:
        sign = 1
    else:
        sign = (-1)**int(m-m_dash)

    mu = np.abs(m_dash - m)
    nu = np.abs(m_dash + m)
    s = n - (mu + nu)/2

    norm = (factorial(s)*factorial(s+mu+nu))/(factorial(s+mu)*factorial(s+nu))
    P = eval_jacobi(s, mu, nu, np.cos(beta))
    d = sign * np.sqrt(norm) * np.sin(beta/2)**mu * np.cos(beta/2)**nu * P

    return d
