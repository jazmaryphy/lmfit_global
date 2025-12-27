# %%
import numpy as np
from scipy.special import erf, erfi, wofz, j0, j1

log2 = np.log(2)
s2pi = np.sqrt(2*np.pi)
s2 = np.sqrt(2.0)
# tiny had been numpy.finfo(numpy.float64).eps ~=2.2e16.
# here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
tiny = 1.0e-15

# %%
def not_zero(value):
    """Return value with a minimal absolute size of tiny, preserving the sign.

    This is a helper function to prevent ZeroDivisionError's.

    Args::
        value (float): Value to be ensured not to be zero.

    Returns:
        float: Value ensured not to be zero.

    """
    return float(np.copysign(max(tiny, abs(value)), value))

# %% [markdown]
# lmfit lineshapes...

# %%
def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * np.exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))


def lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Lorentzian function.

    lorentzian(x, amplitude, center, sigma) =
        (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)

    """
    return ((amplitude/(1 + ((1.0*x-center)/max(tiny, sigma))**2))
            / max(tiny, (np.pi*sigma)))


def step(x, amplitude=1.0, center=0.0, sigma=1.0, form='linear'):
    """Return a step function.

    Starts at 0.0, ends at `sign(sigma)*amplitude`, has a half-max at
    `center`, rising or falling with `form`:

    - `'linear'` (default) = amplitude * min(1, max(0, arg + 0.5))
    - `'atan'`, `'arctan'` = amplitude * (0.5 + atan(arg)/pi)
    - `'erf'`              = amplitude * (1 + erf(arg))/2.0
    - `'logistic'`         = amplitude * [1 - 1/(1 + exp(arg))]

    where ``arg = (x - center)/sigma``.

    Note that ``sigma > 0`` gives a rising step, while ``sigma < 0`` gives
    a falling step.
    """
    out = np.sign(sigma)*(x - center)/max(tiny*tiny, abs(sigma))

    if form == 'erf':
        out = 0.5*(1 + erf(out))
    elif form == 'logistic':
        out = 1. - 1./(1. + np.exp(out))
    elif form in ('atan', 'arctan'):
        out = 0.5 + np.arctan(out)/np.pi
    elif form == 'linear':
        out = np.minimum(1, np.maximum(0, out + 0.5))
    else:
        msg = (f"Invalid value ('{form}') for argument 'form'; should be one "
               "of 'erf', 'logistic', 'atan', 'arctan', or 'linear'.")
        raise ValueError(msg)

    return amplitude*out

def voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None):
    """Return a 1-dimensional Voigt function.

    voigt(x, amplitude, center, sigma, gamma) =
        amplitude*real(wofz(z)) / (sigma*s2pi)

    For more information, see: https://en.wikipedia.org/wiki/Voigt_profile

    """
    if gamma is None:
        gamma = sigma
    z = (x-center + 1j*gamma) / max(tiny, (sigma*s2))
    return amplitude*np.real(wofz(z)) / max(tiny, (sigma*s2pi))


def pvoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    """Return a 1-dimensional pseudo-Voigt function.

    pvoigt(x, amplitude, center, sigma, fraction) =
        amplitude*(1-fraction)*gaussian(x, center, sigma_g) +
        amplitude*fraction*lorentzian(x, center, sigma)

    where `sigma_g` (the sigma for the Gaussian component) is

        ``sigma_g = sigma / sqrt(2*log(2)) ~= sigma / 1.17741``

    so that the Gaussian and Lorentzian components have the same FWHM of
    ``2.0*sigma``.

    """
    sigma_g = sigma / np.sqrt(2*log2)
    return ((1-fraction)*gaussian(x, amplitude, center, sigma_g) +
            fraction*lorentzian(x, amplitude, center, sigma))


def exponential(x, amplitude=1, decay=1):
    """Return an exponential function.

    exponential(x, amplitude, decay) = amplitude * exp(-x/decay)

    """
    decay = not_zero(decay)
    return amplitude * np.exp(-x/decay)


def sine(x, amplitude=1.0, frequency=1.0, shift=0.0):
    """Return a sinusoidal function.

    sine(x, amplitude, frequency, shift) =
        amplitude * sin(x*frequency + shift)

    """
    return amplitude*np.sin(x*frequency + shift)


def expsine(x, amplitude=1.0, frequency=1.0, shift=0.0, decay=0.0):
    """Return an exponentially decaying sinusoidal function.

    expsine(x, amplitude, frequency, shift, decay) =
        amplitude * sin(x*frequency + shift) * exp(-x*decay)

    """
    return amplitude*np.sin(x*frequency + shift) * np.exp(-x*decay)
    

def cosine(x, amplitude=1.0, frequency=1.0, shift=0.0):
    """Return a sinusoidal function.

    sine(x, amplitude, frequency, shift) =
        amplitude * sin(x*frequency + shift)

    """
    return amplitude*np.cos(x*frequency + shift)


def expcosine(x, amplitude=1.0, frequency=1.0, shift=0.0, decay=0.0):
    """Return an exponentially decaying sinusoidal function.

    expsine(x, amplitude, frequency, shift, decay) =
        amplitude * sin(x*frequency + shift) * exp(-x*decay)

    """
    return amplitude*np.cos(x*frequency + shift) * np.exp(-x*decay)


def powerlaw(x, amplitude=1, exponent=1.0):
    """Return the powerlaw function.

    powerlaw(x, amplitude, exponent) = amplitude * x**exponent

    """
    return amplitude * x**exponent


def linear(x, slope=1.0, intercept=0.0):
    """Return a linear function.

    linear(x, slope, intercept) = slope * x + intercept

    """
    return slope * x + intercept


def parabolic(x, a=0.0, b=0.0, c=0.0):
    """Return a parabolic function.

    parabolic(x, a, b, c) = a * x**2 + b * x + c

    """
    return a * x**2 + b * x + c


def napro_sum_gaussian(x, amplitude=1.0, magmom=1.0, sigma=1.0, offset=0.0,
                 dipolar=None, x_target=None):
    r"""
    Compute a summed Gaussian field distribution for multiple dipolar fields.

    Args:
        x (array-like): Independent variable (e.g., magnetic field or frequency).
        amplitude (float, optional): Amplitude or weight of each Gaussian component. 
            Default is 1.0.
        magmom (float, optional): Magnetic moment scaling factor applied to each 
            dipolar field. Default is 1.0.
        sigma (float, optional):Standard deviation (width) of each Gaussian component. 
            Default is 1.0.
        offset (float, optional): Constant background added to the summed distribution. 
            Default is 0.0.
        dipolar (array-like): Array of dipolar field centers. These values are multiplied 
            by ``magmom`` before evaluating the Gaussians.
        x_target (array-like, optional): If provided, the final distribution is interpolated 
            onto this grid.

    Returns:
        ndarray:
            The summed Gaussian field distribution, optionally interpolated
            onto ``x_target``.

    Raises:
        ValueError:
            If ``dipolar`` is not provided.

    """
    if dipolar is None:
        raise ValueError("dipolar must be provided for this calculation.")

    dipolar = np.asarray(dipolar) * magmom

    rho = np.sum(
        [gaussian(x, amplitude=np.abs(amplitude), center=c, sigma=sigma)
         for c in dipolar],
        axis=0
    )

    rho = rho + offset

    if x_target is not None:
        return np.interp(x_target, x, rho)

    return rho

# %% [markdown]
# musrfit type functions...

# %%
def const(t, const=0.0):
    r"""Returns Constant baseline function.

    Args:
        t (array-like): Time values (μs). Only the shape is used.
        const (float): Constant value returned for all time points.
    """
    return np.full_like(t, const)


def asymmetry(t, A=0.0):
    r"""Returns Constant asymmetry function.

    Args:
        t (array-like): Time values (μs). Only the shape is used.
        A (float): Asymmetry amplitude.
    """
    return np.full_like(t, A)


def simplExpo(t, lam=1.0):
    r"""Returns Simple exponential relaxation function.

    Args:
        t (array-like): Time values in microseconds.
        lam (float): Relaxation rate (:math:`\mu\text{s}^{-1}`).

    simplExpo(t, lam) = exp(-lam * t)
    """
    return np.exp(-lam * t)


def generExpo(t, lam=1.0, beta=1.0):
    r"""Returns Generalized (stretched) exponential relaxation function.

    Args:
        t (array-like): Time values in microseconds.
        lam (float): Characteristic relaxation rate (:math:`\mu\text{s}^{-1}`).
        beta (float): Stretching exponent (unitless).

    generExpo(t, lam, beta) = exp(-(lam * t) ** beta)
    """
    return np.exp(-(lam * t) ** beta)


def simpleGss(t, sigma=1.0):
    r"""Returns Simple Gaussian relaxation function.

    Args:
        t (array-like): Time values in microseconds.
        sigma (float): Gaussian relaxation width (:math:`\mu\text{s}^{-1}`).

    simpleGss(t, sigma) = exp(-0.5 * (sigma * t) ** 2)
    """
    return np.exp(-0.5 * (sigma * t) ** 2)


def statGssKT(t, sigma=1.0):
    r"""Returns Static Gaussian Kubo-Toyabe (KT) relaxation function.

    Args:
        t (array-like): Time values in microseconds.
        sigma (float): Field distribution width (:math:`\mu\text{s}^{-1}`).

    statGssKT(t, sigma) =
        (1/3) + (2/3) * (1 - (sigma*t)**2) * exp(-0.5*(sigma*t)**2)
    """
    return (1/3) + (2/3) * (1 - (sigma * t) ** 2) * np.exp(-0.5 * (sigma * t) ** 2)


def statExpKT(t, lam=1.0):
    r"""Returns Static exponential Kubo-Toyabe (KT) relaxation function.

    Args:
        t (array-like): Time values in microseconds.
        lam (float): Exponential relaxation rate (:math:`\mu\text{s}^{-1}`).

    statExpKT(t, lam) = 
        (1/3) + (2/3) * (1 - lam*t) * np.exp(-lam*t)
    """
    return (1/3) + (2/3) * (1 - lam * t) * np.exp(-lam * t)


def dynGLKT_F_LF(t, nu=0.0, sigma=1.0, gamma=1.0):
    r"""Retuns Dynamic Gaussian Longitudinal Kubo-Toyabe in 
    Longitudinal Field (LF) approximation.

    Args:
        t (array-like): Time values in microseconds.
        nu (float): Frequency in MHz.
        sigma (float): Gaussian relaxation width (:math:`\mu\text{s}^{-1}`).
        gamma (float): Fluctuation rate in MHz.
    """
    omega = 2 * np.pi * nu
    exp_gamma_t = np.exp(-gamma * t)
    cos_term = (omega**2 - gamma**2) * (1 - exp_gamma_t * np.cos(omega * t))
    sin_term = -2 * gamma * omega * exp_gamma_t * np.sin(omega * t)
    gamma_t = ((omega**2 + gamma**2) * gamma * t + cos_term + sin_term) / (omega**2 + gamma**2)**2
    return np.exp(-np.sqrt(4 * sigma**2 * gamma_t))


def combiGLKT(t, lam=1.0, sigma=1.0):
    r"""Returns Combined Gaussian and exponential Kubo-Toyabe model.

    Args:
        t (array-like): Time values in microseconds.
        lam (float): Exponential relaxation rate (:math:`\mu\text{s}^{-1}`).
        sigma (float): Gaussian relaxation width (:math:`\mu\text{s}^{-1}`).

    combiGLKT(t, lam, sigma) = 
        (1/3) + (2/3) * (1 - sigma**2*t**2 - lam*t) * exp(-0.5*sigma**2*t**2 - lam*t)
    """
    return (1/3) + (2/3) * (1 - sigma**2*t**2 - lam*t) * np.exp(-0.5*sigma**2*t**2 - lam*t)


def strKT(t, sigma, beta):
    r"""Returns Stretched Kubo-Toyabe model.

    Args:
        t (array-like): Time values in microseconds.
        sigma (float): Gaussian relaxation width (:math:`\mu\text{s}^{-1}`).
        beta (float): Stretching exponent (unitless).

    strKT(t, sigma, beta) = 
        (1/3) + (2/3) * (1 - (sigma*t)**beta) * exp(-((sigma * t)**beta)/beta)
    """
    beta = not_zero(beta)
    return (1/3) + (2/3) * (1 - (sigma * t)**beta) * np.exp(-((sigma * t)**beta) / beta)


def spinGlass(t, lam=1.0, gamma=1.0, q=1.0):
    r"""Returns Spin glass relaxation model.

    Args:
        t (array-like): Time values in microseconds.
        lam (float): Relaxation rate (:math:`\mu\text{s}^{-1}`).
        gamma (float): Fluctuation rate (:math:`\mu\text{s}^{-1}`).
        q (float): Spin glass fraction (unitless).
    """
    gamma = not_zero(gamma)
    omega_t = 4 * lam**2 * (1 - q) * t / gamma
    sqrt_term = np.sqrt(omega_t + q * lam**2 * t**2)
    part1 = (1/3) * np.exp(-np.sqrt(omega_t))
    part2 = (2/3) * (1 - q * lam**2 * t**2 / sqrt_term) * np.exp(-sqrt_term)
    return part1 + part2


def rdAnisoHf(t, nu=0.0, lam=1.0):
    r"""Returns Random anisotropic hyperfine field model.

    Args:
        t (array-like): Time values in microseconds.
        nu (float): Frequency in MHz.
        lam (float): Relaxation rate (:math:`\mu\text{s}^{-1}`).
    """
    term1 = (1/6) * (1 - nu * t / 2) * np.exp(-nu * t / 2)
    term2 = (1/3) * (1 - nu * t / 4) * np.exp(-nu * t / 4.44994)
    return term1 + term2


def TFieldCos(t, phi=0.0, nu=0.0):
    r"""Returns Transverse field cosine oscillation model.

    Args:
        t (array-like): Time values in microseconds.
        phi (float): Phase in degrees.
        nu (float): Frequency in MHz.

    TFieldCos(t, phi, nu) = 
        np.cos(2 * np.pi * nu * t + np.pi * phi / 180)
    """
    return np.cos(2 * np.pi * nu * t + np.pi * phi / 180)


def internFld(t, alpha=1.0, phi=0.0, nu=0.0, lam_T=1.0, lam_L=1.0):
    r"""Returns Internal field model with transverse and longitudinal relaxation.

    Args:
        t (array-like): Time values in microseconds.
        alpha (float): Transverse fraction (:math:`0 \le \alpha \le 1`).
        phi (float): Phase in degrees.
        nu (float): Frequency in MHz.
        lam_T (float): Transverse relaxation rate (:math:`\mu\text{s}^{-1}`).
        lam_L (float): Longitudinal relaxation rate (:math:`\mu\text{s}^{-1}`).
    """
    phase = 2 * np.pi * nu * t + np.pi * phi / 180
    transverse = alpha * np.cos(phase) * np.exp(-lam_T * t)
    longitudinal = (1 - alpha) * np.exp(-lam_L * t)
    return transverse + longitudinal


def Bessel(t, phi=0.0, nu=0.0):
    r"""Returns Bessel function model.

    Args:
        t (array-like): Time values in microseconds.
        phi (float): Phase in degrees.
        nu (float): Frequency in MHz.

    Bessel(t, phi, nu) = 
        j0(2 * np.pi * nu * t + np.pi * phi / 180)
    """
    return j0(2 * np.pi * nu * t + np.pi * phi / 180)


def internbsl(t, alpha=1.0, phi=0.0, nu=0.0, lam_T=1.0, lam_L=1.0):
    r"""Returns Internal field model with Bessel modulation and exponential relaxation.

    Args:
        t (array-like): Time values in microseconds.
        alpha (float): Transverse fraction (:math:`0 \le \alpha \le 1`).
        phi (float): Phase in degrees.
        nu (float): Frequency in MHz.
        lam_T (float): Transverse relaxation rate (:math:`\mu\text{s}^{-1}`).
        lam_L (float): Longitudinal relaxation rate (:math:`\mu\text{s}^{-1}`).
    """
    phase = 2 * np.pi * nu * t + np.pi * phi / 180
    transverse = alpha * j0(phase) * np.exp(-lam_T * t)
    longitudinal = (1 - alpha) * np.exp(-lam_L * t)
    return transverse + longitudinal


def internFldGK(t, alpha=1.0, nu=1.0, sigma=1.0, lam=1.0, beta=1.0):
    r"""Returns Internal field model with Gaussian KT and stretched exponential.

    Args:
        t (array-like): Time values in microseconds.
        alpha (float): Transverse fraction (:math:`0 \le \alpha \le 1`).
        nu (float): Frequency in MHz.
        sigma (float): Gaussian relaxation width (:math:`\mu\text{s}^{-1}`).
        lam (float): Exponential relaxation rate (:math:`\mu\text{s}^{-1}`).
        beta (float): Stretching exponent (unitless).
    """
    nu = not_zero(nu)
    omega = 2 * np.pi * nu
    cos_term = np.cos(omega * t)
    sin_term = np.sin(omega * t)
    gkt = (cos_term - (sigma**2 * t**2 / omega) * sin_term) * np.exp(-0.5 * sigma**2 * t**2)
    stretched = np.exp(-(lam * t) ** beta)
    return alpha * gkt + (1 - alpha) * stretched


def internFldLL(t, alpha=1.0, nu=1.0, a=1.0, lam=1.0, beta=1.0):
    r"""Returns Internal field model with Lorentzian modulation and stretched exponential.

    Args:
        t (array-like): Time values in microseconds.
        alpha (float): Transverse fraction (:math:`0 \le \alpha \le 1`).
        nu (float): Frequency in MHz.
        a (float): Damping coefficient a (:math:`\mu\text{s}^{-1}`).
        lam (float): Exponential relaxation rate (:math:`\mu\text{s}^{-1}`).
        beta (float): Stretching exponent (unitless).
    """
    omega = 2 * np.pi * nu
    cos_term = np.cos(omega * t)
    sin_term = np.sin(omega * t)
    lorentz = (cos_term - (a / omega) * sin_term) * np.exp(-a * t)
    stretched = np.exp(-(lam * t) ** beta)
    return alpha * lorentz + (1 - alpha) * stretched


def F_mu_F(t, omega_d=0.0):
    r"""Returns Muon spin polarization function for F-:math:`\mu`-F systems.

    Args:
        t (array-like): Time values in microseconds.
        omega_d (float): Dipolar frequency math:`\omega_d` in MHz.
    """
    term1 = np.cos(np.sqrt(3) * omega_d * t)
    term2 = (1 - 1 / np.sqrt(3)) * np.cos(((3 - np.sqrt(3)) / 2) * omega_d * t)
    term3 = (1 + 1 / np.sqrt(3)) * np.cos(((3 + np.sqrt(3)) / 2) * omega_d * t)
    return (1/6) * (3 + term1 + term2 + term3)


def fmuf(t, omega_d=0.0):
    r"""Returns Muon spin polarization function for F-:math:`\mu`-F systems.

    Args:
        t (array-like): Time values in microseconds.
        omega_d (float): Dipolar frequency math:`\omega_d` in MHz.
    """
    term1 = np.cos(np.sqrt(3) * omega_d * t)
    term2 = (1 - 1 / np.sqrt(3)) * np.cos(((3 - np.sqrt(3)) / 2) * omega_d * t)
    term3 = (1 + 1 / np.sqrt(3)) * np.cos(((3 + np.sqrt(3)) / 2) * omega_d * t)
    return (1/6) * (3 + term1 + term2 + term3)


def abragam(t, sigma=1.0, gamma=1.0):
    r"""Returns Abragam relaxation function.

    Args:
        t (array-like): Time values in microseconds.
        sigma (float): Gaussian field width (:math:`\mu\text{s}^{-1}`).
        gamma (float): Fluctuation rate (MHz).

    abragam(t, sigma, gamma) = 
        exp(-(sigma**2/gamma**2) * (exp(-gamma*t) - 1 + gamma*t))
    """
    gamma = not_zero(gamma)
    return np.exp(-(sigma**2 / gamma**2) * (np.exp(-gamma * t) - 1 + gamma * t))


def skewedGss(t, phi, nu, sigma_plus, sigma_minus):
    r"""Returns Skewed Gaussian relaxation function with asymmetric field distributions.

    Args:
        t (array-like): Time values in microseconds.
        phi_deg (float): Phase in degrees.
        nu (float): Frequency in MHz.
        sigma_plus (float): Gaussian width (:math:`\mu\text{s}^{-1}`).
        sigma_minus (float): Gaussian width (:math:`\mu\text{s}^{-1}`).
    """
    phase = 2 * np.pi * nu * t + np.pi * phi / 180
    norm = sigma_plus + sigma_minus

    term_minus = (sigma_minus / norm) * np.exp(-0.5 * sigma_minus**2 * t**2)
    term_minus *= np.cos(phase) + np.sin(phase) * erfi(sigma_minus * t / np.sqrt(2))

    term_plus = (sigma_plus / norm) * np.exp(-0.5 * sigma_plus**2 * t**2)
    term_plus *= np.cos(phase) - np.sin(phase) * erfi(sigma_plus * t / np.sqrt(2))

    return term_minus + term_plus


def staticNKZF(t, delta0=1.0, Rb=0.0):
    r"""Returns Static Narrow Kubo-Toyabe Zero Field model.

    Args:
        t (array-like): Time values in microseconds.
        delta0 (float): field distribution width (:math:`\mu\text{s}^{-1}`).
        Rb (float): Ratio :math:`\Delta_{\mathrm{GbG}} / \Delta_0` (unitless).
    """
    x = delta0 * t
    denom = 1 + (Rb * x) ** 2
    return (1/3) + (2/3) * denom**(-1.5) * (1 - x**2 / denom) * np.exp(-x**2 / (2 * denom))


def staticNKTF(t, phi=0.0, nu=0.0, delta0=1.0, Rb=0.0):
    r"""Returns  Static Narrow Kubo-Toyabe Transverse Field model.

    Args:
        t (array-like): Time values in microseconds.
        phi_deg (float): Phase in degrees.
        nu (float): Frequency in MHz.
        delta0 (float): field distribution width (:math:`\mu\text{s}^{-1}`).
        Rb (float): Ratio :math:`\Delta_{\mathrm{GbG}} / \Delta_0` (unitless).

    Returns:
        ndarray: Relaxation function values.
    """
    x = delta0 * t
    denom = 1 + (Rb * x) ** 2
    envelope = np.sqrt(1 / denom) * np.exp(-x**2 / (2 * denom))
    phase = 2 * np.pi * nu * t + np.pi * phi / 180
    return envelope * np.cos(phase)


def dynamicNKZF(t, delta0=1.0, Rb=0.0, nuc=1.0):
    r"""Returns Dynamic Narrow Kubo-Toyabe Zero Field model.

    Args:
        t (array-like): Time values in microseconds.
        delta0 (float): field distribution width (:math:`\mu\text{s}^{-1}`).
        Rb (float): Ratio :math:`\Delta_{\mathrm{GbG}} / \Delta_0` (unitless).
        nuc (float): Fluctuation rate in MHz.
    """
    nuc = not_zero(nuc)
    theta = (np.exp(-nuc * t) - 1 + nuc * t) / nuc**2
    denom = 1 + Rb**2 * delta0**2 * theta
    return np.sqrt(1 / denom) * np.exp(-2 * delta0**2 * theta / denom)


def dynamicNKTF(t, phi=0.0, nu=0.0, delta0=1.0, Rb=0.0, nuc=1.0):
    r"""Returns Dynamic Narrow Kubo-Toyabe Transverse Field model.

    Args:
        t (array-like): Time values in microseconds.
        phi_deg (float): Phase in degrees.
        nu (float): Frequency in MHz.
        delta0 (float): field distribution width (:math:`\mu\text{s}^{-1}`).
        Rb (float): Ratio :math:`\Delta_{\mathrm{GbG}} / \Delta_0` (unitless).
        nuc (float): Fluctuation rate in MHz.
    """
    nuc = not_zero(nuc)
    theta = (np.exp(-nuc * t) - 1 + nuc * t) / nuc**2
    denom = 1 + Rb**2 * delta0**2 * theta
    envelope = np.sqrt(1 / denom) * np.exp(-delta0**2 * theta / denom)
    phase = 2 * np.pi * nu * t + np.pi * phi / 180
    return envelope * np.cos(phase)


def muMinusExpTF(t, N0=1.0, tau=1.0, A=0.0, lam=1.0, phi=0.0, nu=0.0):
    r"""Returns Mu-minus exponential transverse field model.

    Args:
        t (array-like): Time values in microseconds.
        N0 (float): Initial normalization factor.
        tau (float): Muon lifetime (:math:`\mu\text{s}`).
        A (float): Asymmetry amplitude.
        lam (float): Relaxation rate (:math:`\mu\text{s}^{-1}`).
        phi (float): Phase in degrees.
        nu (float): Frequency in MHz.

    Returns:
        ndarray: Relaxation function values.
    """
    tau = not_zero(tau)
    decay = np.exp(-t / tau)
    envelope = A * np.exp(-lam * t)
    phase = 2 * np.pi * nu * t + np.pi * phi / 180
    return N0 * decay * (1 + envelope * np.cos(phase))


def polynom(t, t0=0.0, *a):
    """Returns Polynomial model centered at t₀.

    Args:
        t (array-like): Time values.
        t0 (float): Center time for polynomial expansion.
        *a (float): Coefficients a0, a1, ..., an.
    """
    dt = t - t0
    return sum(a_k * dt**k for k, a_k in enumerate(a))


