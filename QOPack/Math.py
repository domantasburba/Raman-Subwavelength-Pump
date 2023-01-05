import numba
from numba import njit
import numpy as np
from scipy.interpolate import splrep, splev


@numba.vectorize([numba.float64(numba.complex128)])
def abs2(z):
    return z.real**2 + z.imag**2


@numba.vectorize([numba.float64(numba.complex128)])
def abs4(z):
    return (z.real**2 + z.imag**2)**2


@njit(cache=True)
def Gaussian(x_arr, mu, sigma):
    return np.exp(-0.5 * ((x_arr - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


@njit(cache=True)
def Pauli_x():
    return np.array([
        [0., 1.],
        [1., 0.]
    ], dtype=np.complex128)


@njit(cache=True)
def Pauli_y():
    return np.array([
        [0., -1j],
        [1j, 0.]
    ], dtype=np.complex128)


@njit(cache=True)
def Pauli_z():
    return np.array([
        [1., 0.],
        [0., -1.]
    ], dtype=np.complex128)


# @njit(cache=True)
def Pauli_vector():
    sigma_x = Pauli_x()
    sigma_y = Pauli_y()
    sigma_z = Pauli_z()

    return np.array([
        sigma_x,
        sigma_y,
        sigma_z
    ])


def calc_step(x_span, N_x):
    """Assumes equidistant array spacing."""
    return (x_span[1] - x_span[0]) / (N_x - 1)


def x2index(x_span, N_x, select_x):
    """Assumes equidistant array spacing."""
    if not x_span[0] <= select_x <= x_span[1]:
        print("WARNING: x2index's select_x is out of bounds.")
    return np.int64(np.round((select_x - x_span[0]) / (x_span[1] - x_span[0]) * (N_x - 1)))


def calc_orthogonal_vectors_3D(v):
    v_norm = v / np.sqrt(np.dot(v, v))

    x = np.cross(v_norm, np.array([1.0, 0.0, 0.0]))
    if np.dot(x, x) < 1e-6:
        x = np.cross(v_norm, np.array([0.0, 1.0, 0.0]))
    x = x / np.sqrt(np.dot(x, x))

    y = np.cross(v_norm, x)
    y = y / np.sqrt(np.dot(y, y))

    return x, y


def spectral_Dn_y(y_arr, x_arr, der=1):
    """Assumes:
    1) y_arr is a 1D numpy array;
    2) len(x_arr) = len(y_arr);"""
    dx = (x_arr[-1] - x_arr[0]) / (len(x_arr) - 1)
    k_arr = 2*np.pi*np.fft.fftfreq(len(x_arr), d=dx)

    y_bar = np.fft.fft(y_arr)
    Dn_y_bar = (1j*k_arr)**der * y_bar
    Dn_y = np.fft.ifft(Dn_y_bar)

    if y_arr.dtype == "float64":
        check_if_real(Dn_y, 'Dn_y (from %s)' % spectral_Dn_y.__name__)
        Dn_y = np.real(Dn_y).astype(np.float64)

    return Dn_y


def spl_Dn_y(y_arr, old_x_arr, new_x_arr=None, der=0):
    """Assumes y_arr is either:
    1) a 1D numpy array of dtype float64 or complex128 and that the last axis is the x axis;
    2) a ND numpy array of dtype float64 or complex128 and that the first axis is the x axis;
    If new_x_arr=None, then it is set equal to old_x_arr (discretization isn't changed)."""
    y_dtype = y_arr.dtype

    # By default assume that discretization remains the same.
    if new_x_arr is None:
        new_x_arr = old_x_arr

    y_shape = y_arr.shape
    if len(y_shape) == 1:
        if y_dtype == "float64":
            tck_y = splrep(old_x_arr, y_arr)
            Dn_y = splev(new_x_arr, tck_y, der=der)
        elif y_dtype == "complex128":
            tck_real = splrep(old_x_arr, np.real(y_arr))
            tck_imag = splrep(old_x_arr, np.imag(y_arr))
            Dn_y_real = splev(new_x_arr, tck_real, der=der)
            Dn_y_imag = splev(new_x_arr, tck_imag, der=der)
            Dn_y = Dn_y_real + 1.j * Dn_y_imag
        else:
            # print("WARNING: Unsupported dtype. (from %s)" % spl_Dn_y.__name__")
            raise TypeError(r"Unsupported dtype. (from %s)" % spl_Dn_y.__name__)
    elif len(y_shape) > 1:
        other_dim = 1
        for dim in y_shape[1:]:
            other_dim *= dim

        temp_y_arr = np.reshape(y_arr, (y_shape[0], other_dim))
        Dn_y = np.empty((len(new_x_arr), other_dim), dtype=y_dtype)
        if y_dtype == "float64":
            for idx in range(other_dim):
                tck_y = splrep(old_x_arr, temp_y_arr[:, idx])
                Dn_y[:, idx] = splev(new_x_arr, tck_y, der=der)
        elif y_dtype == "complex128":
            for idx in range(other_dim):
                tck_real = splrep(old_x_arr, np.real(temp_y_arr[:, idx]))
                tck_imag = splrep(old_x_arr, np.imag(temp_y_arr[:, idx]))
                Dn_y_real = splev(new_x_arr, tck_real, der=der)
                Dn_y_imag = splev(new_x_arr, tck_imag, der=der)
                Dn_y[:, idx] = Dn_y_real + 1.j * Dn_y_imag
        else:
            raise TypeError(r"Unsupported dtype. (from %s)" % spl_Dn_y.__name__)

        Dn_y = np.reshape(Dn_y, (len(new_x_arr), *y_shape))
    else:
        raise ValueError(r"Unsupported object. (from %s)" % spl_Dn_y.__name__)

    return Dn_y


@njit(cache=True)
# Taken from https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array.
def translate_y(untranslated_y_arr, x_span, x_shift):
    """
    Returns translated copy of given function.

    Assumes the following:
    * 1D real space;
    * last axis corresponds to spatial dimension (shifts along that dimension);
    * given function is zero outside its bounds;

    Parameters
    ----------
    untranslated_y_arr: array
        Function, whose copy will be translated.
    x_span: tuple of size 2
        The beginning and end of our real space.
    x_shift: float
        Translation distance. If positive, moves function to the right; if negative, moves function to the left.

    Returns
    -------
    translated_y_arr: array
        Translated copy of given function.
    """
    N_x = np.shape(untranslated_y_arr)[-1]
    shift_index = np.int64(np.round(x_shift / (x_span[1] - x_span[0]) * (N_x - 1)))

    translated_y_arr = np.zeros_like(untranslated_y_arr)
    # if np.shape(np.shape(untranslated_y_arr)) == (1,):
    #     if shift_index > 0:
    #         translated_y_arr[shift_index:] = untranslated_y_arr[:-shift_index]
    #     elif shift_index < 0:
    #         translated_y_arr[:shift_index] = untranslated_y_arr[-shift_index:]
    #     else:
    #         translated_y_arr[:] = untranslated_y_arr
    # else:
    if shift_index > 0:
        translated_y_arr[..., shift_index:] = untranslated_y_arr[..., :-shift_index]
    elif shift_index < 0:
        translated_y_arr[..., :shift_index] = untranslated_y_arr[..., -shift_index:]
    else:
        translated_y_arr[...] = untranslated_y_arr

    return translated_y_arr


def normalize_wavefunction(unnormalized_psi_arr, x_arr, dtype=np.complex128):
    """Assumes that last axis is spatial axis and 1D real space."""
    # psi_abs2_integral = np.trapz(abs2(unnormalized_psi_arr), x_arr, axis=-1)
    psi_abs2_integral = np.trapz((np.abs(unnormalized_psi_arr))**2, x_arr, axis=-1)
    normalized_psi_arr = unnormalized_psi_arr / np.sqrt(psi_abs2_integral)[..., np.newaxis]

    return normalized_psi_arr.astype(dtype)


# @njit(cache=True)
# TODO: Make compatible with Numba
def normalize_ket(unnormalized_ket_arr, dtype=np.complex128):
    """Assumes that last axis is component axis."""
    bra_ket = np.sum(abs2(unnormalized_ket_arr), axis=-1)
    normalized_ket_arr = unnormalized_ket_arr / np.expand_dims(np.sqrt(bra_ket), axis=-1)

    return normalized_ket_arr.astype(dtype)


def check_if_real(arr, name='arr'):
    # assert np.all(np.isclose(np.imag(arr), np.zeros(np.shape(arr), dtype=np.float64))), "%s isn't real." % name
    # np.logical_not(np.all(np.isclose(np.imag(arr), np.zeros(np.shape(arr), dtype=np.float64)))) or print("WARNING: %s isn't real." % name)
    # if np.any(np.logical_not(np.isclose(np.imag(arr), np.zeros(np.shape(arr), dtype=np.float64)))):
    if not np.allclose(np.imag(arr), np.zeros(np.shape(arr), dtype=np.float64)):
        print("WARNING: %s isn't real." % name)


def check_if_imag(arr, name='arr'):
    if not np.allclose(np.real(arr), np.zeros(np.shape(arr), dtype=np.float64)):
        print("WARNING: %s isn't imaginary." % name)


def check_if_int(arr, name='arr'):
    # if np.any(np.logical_not(np.isclose(arr, np.int64(np.round(np.real(arr)))))):
    if not np.allclose(arr, np.int64(np.round(np.real(arr)))):
        print("WARNING: %s isn't an integer." % name)


def check_if_Hermitian(arr, name='arr'):
    arr_hermiticity = np.sum(abs2(arr - np.transpose(np.conj(arr))))

    epsilon = 1E-5
    if arr_hermiticity > epsilon:
        print("WARNING: %s isn't Hermitian." % name)


def check_if_close(arr1, arr2, name1='arr1', name2='arr2'):
    # if np.any(np.logical_not(np.isclose(arr1, arr2))):
    if not np.allclose(arr1, arr2):
        print("WARNING: %s and %s aren't close to equal." % (name1, name2))


def check_ket_normalization(ket, name='ket'):
    """Assumes last axis is component axis."""
    ket_norm = np.sum(abs2(ket), axis=-1)

    ones_arr = np.ones(ket[..., 0].shape)

    if not np.allclose(ket_norm, ones_arr):
        print("WARNING: %s isn't normalized." % name)


def check_ket_orthonorm(ket, name='ket'):
    """Assumes first axis is orthonorm axis and last axis is component axis."""
    ket_orthonorm = np.sum(np.conj(ket[np.newaxis, ...]) * ket[:, np.newaxis, ...], axis=-1)

    N_orthonorm = ket.shape[0]
    ones_arr = [1 for i in range(len(ket.shape) - 2)]
    identity_arr = np.identity(N_orthonorm).reshape(N_orthonorm, N_orthonorm, *ones_arr)

    # if np.any(np.logical_not(np.isclose(ket_orthonorm, identity_arr))):
    if not np.allclose(ket_orthonorm, identity_arr):
        print("WARNING: %s isn't orthonormal." % name)
