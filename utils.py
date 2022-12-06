import scipy
import numpy as np

def Gaussian(x, N, FWHM, mu):
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    Gauss = N * 1/ (sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2 * sigma**2))
    return Gauss

def Lorentzian(x, N, FWHM, mu):
    Lorentz = N * 1/(np.pi) * (FWHM/2) / ((x-mu)**2 + (FWHM/2)**2)
    return Lorentz

def Voigt(x, N, G_FWHM, L_FWHM, mu):
    sigma = G_FWHM/(2*np.sqrt(2*np.log(2)))
    L_HWHM = L_FWHM/2 # Scipy's voigt_profile expects Half Width at Half Max as the Lorentzian width parameter
    return N * scipy.special.voigt_profile(x-mu, sigma, L_HWHM)
