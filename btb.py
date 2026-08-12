# Briefer than brief (BTB) stimuli module 

import numpy as np
from scipy.fft import fft, ifft
from scipy.special import factorial
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

def generate_btb(stim, times, vir_tau=0.005, speedup=2, frame_rate=480):
    ''' Generates stimulus that produces briefer-than-brief response

    Parameters
    ----------
    stim: array-like
        The original stimulus that will be transformed
    times: array-like
        The time dimension corresponding to stim 
        (must have same length as stim)
    vir_tau: float
        The timescale (tau) of the visual impulse response (VIR) 
        in the Adelson & Berger formulation
    speedup: float
        The factor by which to speed up the VIR and thereby 
        the target response
    frame_rate: float
        The frame rate of the screen (in Hz) to present the stimulus on; 
        mainly serves as a scaling factor

    Returns
    -------
    stim_btb: array
        The briefer-than-brief stimulus, which produces a visual response
        that is temporally less smeared out (by a factor of speedup) 
        than the visual response to the original stimulus (stim)
    '''
    
    # assumed real VIR, with vir_tau as timescale parameter
    vir = adelson_bergen_vir(times, tau = vir_tau, n=2) / frame_rate
    vir = np.roll(vir, int(len(times)/2)) # center to prevent edge artifacts
    vir_f = fft(vir) # take fast fourier transform 

    # sped up VIR, with vir_tau/speedup as timescale parameter
    vir_prime = adelson_bergen_vir(times, tau = vir_tau/speedup, n=2) / frame_rate
    vir_prime = np.roll(vir_prime, int(len(times)/2))

    # estimated sped up response through convolution, 
    # which becomes our target response
    stim_prime = np.convolve(stim, vir_prime, mode='same') 
    stim_prime_f = fft(stim_prime) # take fast fourier transform

    # do inverse filtering by dividing the target response
    # by the original impulse response in the fourier domain (_f)
    # and convert back to the time domain
    stim_btb = ifft(stim_prime_f / vir_f).real
    stim_btb = np.roll(stim_btb, int(len(times)/2)-1)
    
    # scaling factor ensures normalization around 0.5, within limits 0, 1
    scaling_factor = np.max(abs(stim_btb)) * 2
    
    return stim_btb, scaling_factor

def adelson_bergen_vir(t, tau=0.005, n=2, nu=0.):
    ''' Returns visual impulse response according to the Adelson & Bergen formulation

    Parameters
    ----------
    t: array
        The time dimension
    tau: float
        The timescale of the impulse response
    n: int
        Parameter controlling the shape of the impulse response
    nu: float
        Magnitude of the negative lobe of the impulse response 

    Returns 
    -------
    vir: array
        The visual impulse response function (VIR)
    '''
    vir = 1/(tau*(1 - nu))*np.exp(-t/tau)*((t/tau)**n/factorial(n) - \
                                           nu*(t/tau)**(n + 2)/factorial(n + 2))
    return vir

def generate_stim(times, isi, pulse_duration=4, sigma=1.5):
    ''' Generates temporally blurred flashes with inter-stimulus interval (ISI)

    Parameters
    ----------
    times: array-like
        The time dimension corresponding to stim 
        (stim will have same length as times)
    isi: int
        The inter-stimulus interval (isi) between the two flashes
    pulse_duration: int
        The duration of the individual flashes
    sigma: float (positive)
        The standard deviation of the Gaussian temporal blur
        which is necessary to avoid high frequencies dominating
        in the briefer-than-brief calculations

    Returns
    -------
    stim: array
        The (original) visual stimulus 
    '''
    
    # isi and pulse duration in samples
    stim = np.concatenate([np.zeros(int(len(times)/2)-pulse_duration-isi), 
                           np.ones(pulse_duration), 
                           np.zeros(isi), 
                           np.ones(pulse_duration),
                           np.zeros(int(len(times)/2)-pulse_duration)])
    stim = gaussian_filter1d(stim, sigma=sigma)
    return stim

def generate_times(stim_dur=0.5, frame_rate=480):
    '''generates time dimension'''
    return np.linspace(0, stim_dur, round(stim_dur*frame_rate))

def visual_response(s, a, b, frame_rate=480):
    '''Computes the visual response given a stimulus (s) and parameters of the VIR: tau (a) and nu (b)
    (convenience function for fitting the VIR given a stimulus and visual response)

    Parameters
    ----------
    s: array-like
        the stimulus
    a: float (positive only)
        VIR parameter tau
    b: float
        VIR parameter nu
    frame_rate: int
        frame rate of the screen in Hz

    Returns
    -------
    resp: array
        the visual response to the stimulus (s convolved with v)
    '''
    t = np.linspace(0, 0.5, round(0.5*frame_rate))
    
    v = adelson_bergen_vir(t, tau=a, nu=b) / frame_rate
    v = np.roll(v, int(len(t)/2))
    resp = np.convolve(s, v, mode='same')
    return resp

def fit_visual_response(stim, btb_response):
    ''' Fits parameters of the VIR given a stimulus and a response

    Parameters
    ----------
    stim: array-like
        the original stimulus (NOT the btb stimulus)
    btb_response: int
        The visual response to the btb stimulus (i.e., btb stimulus
        with the true VIR)

    Returns
    -------
    popt: 2d array
        estimated VIR parameters (tau and nu)
    '''
    popt, pcov = curve_fit(visual_response, stim, btb_response, 
                          p0 = [0.005, 0], bounds=([0,0], ['Inf','Inf']),
                          maxfev=5000)
    return popt