import spharpy.spherical as _sph
import numpy as np
from spharpy.ambi import AmbisonicsSignal
from spharpy._deprecation import convert_coordinates

from numpy.linalg import pinv
from pyfar import Signal

def sht(signal, coordinates, N, sh_kind="real", domain=None, axis=0, mode='least-squares'):
    """Compute the spherical harmonics transform at a certain order N

    Parameters
    ----------
    signal: Signal
        the signal for which the spherical harmonics transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinates on which the signal has been sampled
    N: integer
        Spherical harmonic order
    sh_kind: str
        Use real or complex valued SH bases ``'real'``, ``'complex'``
        default is ``'real'``
    domain: str
        Time-frequency domain on which the SH transform is computed 
        default is ``'time'``
    axis: integer
        Axis along which the SH transform is computed 
    mode: 
        ``'least-squares'``, ``'tikhonov'``
        default is ``'least-squares'``
    """

    if domain == None:
        domain = signal.domain

    coordinates = convert_coordinates(coordinates)
    
    if not signal.cshape[axis] == coordinates.n_points: 
        if not coordinates.n_points in signal.cshape:
            raise ValueError("signal shape doesnt match number of coordinates")
        else:
            axis = signal.cshape.index(coordinates.n_points)
            Warning(f"compute SHT along axis={axis}")

    if sh_kind == 'real':
        Y = _sph.spherical_harmonic_basis_real(N, coordinates)
    elif sh_kind == 'complex':
        Y = _sph.spherical_harmonic_basis(N, coordinates)
    else:
        raise ValueError(f"sh_kind should be ``'real'`` or ``'complex'`` but is {sh_kind}")
        
    if domain == "time":
        data = signal.time
    elif domain == "freq":
        data = signal.freq
    else:
        raise ValueError(f"domain should be ``'time'`` or ``'freq'`` but is {domain}")
    
    if mode == 'least-squares':
        data_nm = np.tensordot(pinv(Y), data, [1, axis])
    else:
        raise ValueError('{mode} is not implemented so far')
    
    return AmbisonicsSignal(data=data_nm, domain=domain, sh_kind=sh_kind, 
                            sampling_rate=signal.sampling_rate, fft_norm=signal.fft_norm, 
                            comment=signal.comment)

def isht(ambisonics_signal, coordinates):
    """Compute the spherical harmonics transform at a certain order N

    Parameters
    ----------
    ambisonics_signal: Signal
        The ambisonics signal for which the inverse spherical harmonics transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinates for which the inverse SH transform is computed
    """
    
    coordinates = convert_coordinates(coordinates)
    
    if ambisonics_signal.sh_kind == 'real':
        Y = _sph.spherical_harmonic_basis_real(ambisonics_signal.N, coordinates)
    else:
        Y = _sph.spherical_harmonic_basis(ambisonics_signal.N, coordinates)
    
    if ambisonics_signal.domain == "time":
        data_nm = ambisonics_signal.time
    else:
        data_nm = ambisonics_signal.freq
        
    data = np.tensordot(Y, data_nm, [1, 0]) # assume ambisonics channels on first axis
        
    return Signal(data, ambisonics_signal.sampling_rate, fft_norm=ambisonics_signal.fft_norm, 
                  comment=ambisonics_signal.comment)