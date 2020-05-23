import numpy as np
import math
import scipy.signal as sig
import scipy.fftpack as fftp
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance

# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift, samplingrate)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate, False)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift, sampling_rate):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    # Comment added by us vv
    ''' The sample rate is 20kHz or 20 000 samples per second
    If we want to know how many samples fit in 2 ms (window length), 
    we calculate 20 000 * 0.02 which is equal to 400 samples per window.

    N is going to be the number of "windows" overlapping that we can fit using
    all the samples, which is given by round((samples.size - winshift) / step).
    This is because each time we add a new window to the array, we use 
    #winshift new samples, except for the first one where we use #winlen new samples.
    '''
    # Comment added by us ^^

    step = winlen - winshift
    N = math.floor((samples.size - winshift) / step)

    frames = np.zeros((N, winlen))

    for i in range(N):
      frames[i] = samples[step*i : step*i + winlen]
    
    return frames
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    # Comment added by us vv
    ''' a_0 = 1, b_0 = 1 and b_1 = -p
    The rest of b_i and a_j are assumed to be 0
    '''
    # Comment added by us ^^

    return sig.lfilter([1, -p], [1], input)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    # Comment added by us vv
    ''' sym is False to generate periodic window and non-symmetric
    '''
    # Comment added by us ^^

    M = input.shape[1]
    window = sig.hamming(M, False)
    return (input * window)

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    return np.square(np.absolute(fftp.fft(input, nfft)))

def logMelSpectrum(input, samplingrate, plot_fb=False):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    # Comment added by us vv
    '''
    Plot of filter_bank on liner frequency scale included
    '''
    # Comment added by us ^^

    nfft = input.shape[1]
    filter_bank = trfbank(samplingrate, nfft)
    if plot_fb == True:
      plot_filter_bank(filter_bank)

    mel_spec = np.log(input @ filter_bank.T)
    return mel_spec

def plot_filter_bank(f_b):
  # Comment added by us vv
    '''
    FFT components taken to generate higher res freq for low freq of FFT componenets and low res freq for high freq of
    FFT components.
    '''
  # Comment added by us ^^
    plt.xlim(0,180)   
    for i in range(len(f_b)):
      plt.plot(f_b[i])
    plt.show()

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    # Comment added by us vv
    '''
    type =  2 is default value for DCT function
    We dont pass ncep in the dct function but take the first ncep(13) values from the resulting arrays because
    if we pass ncep(13) in the dct function, it truncates the input if value of ncep is smaller than the input length(which it
    is in our case). 
    Note : There is a canvas dicussion about this as well!
    '''
    # Comment added by us ^^
    type = 2
    mfcc_coffs = fftp.dct(input, type)
    return mfcc_coffs[:, 0:nceps]

#Implementation of DTW using a library
def dynamic_type_warping(x, y, dist):
    from dtw import dtw
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    d, LD, AD, path = dtw(x, y, dist)
    return d

def gmm_posteriors(feature_array, utterances, components):
  # Plot GMM posteriors
  print("GMM Posteriors for digit 7 \n")
  model = GaussianMixture(components)
  model.fit(feature_array)

  posteriors = []
  for utterance in utterances:
      posteriors.append(model.predict_proba(utterance[0]))

  for i in range(len(utterances)):
    print(utterances[i][1]['gender'] + " uttering " + utterances[i][1]['digit'] + " for the " + utterances[i][1]['repetition'] + " repetition.:")
    plot_colormesh(posteriors[i])
    # plt.imshow(posteriors[i], interpolation='nearest')
    # plt.show()

def plot_lmfcc(i, d, lmfcc):
  # Plot LMFCC
  if(d['repetition'] == "a"):
    rep = "1st"
  else:
    rep = "2nd"
  print(i)
  print(d['gender'] + " uttering " + d['digit'] + " for the " + rep + " time.:")
  plot_colormesh(lmfcc)

def plot_corelations(feature_array, example_mspec):
  # Comment added by us vv
  '''
  rowvar = False means that each column(coffecient) is considered as a variable i.e corelation b/w coffs is calculated
  if rowvar = True, then corealtion b/w feature vectors would be calculated
  '''
  # Comment added by us ^^
  print('LMFCCs Correlation Coffs - Uncorrelated')
  corelation_coffs = np.corrcoef(feature_array, rowvar=False)
  plot_colormesh(corelation_coffs)

  print('MSPEC(Mel Filterbank) Correlation Coffs - Correlated')
  corelation_coffs_mspec = np.corrcoef(example_mspec, rowvar=False)
  plot_colormesh(corelation_coffs_mspec)

def tidigit2labels(tidigitsarray):
    """
    Return a list of labels including gender, speaker, digit and repetition information for each
    utterance in tidigitsarray. Useful for plots.
    """
    labels = []
    nex = len(tidigitsarray)
    for ex in range(nex):
        labels.append(tidigitsarray[ex]['gender'] + '_' + 
                      tidigitsarray[ex]['speaker'] + '_' + 
                      tidigitsarray[ex]['digit'] + '_' + 
                      tidigitsarray[ex]['repetition'])
    return labels

def dither(samples, level=1.0):
    """
    Applies dithering to the samples. Adds Gaussian noise to the samples to avoid numerical
        errors in the subsequent FFT calculations.

        samples: array of speech samples
        level: decides the amount of dithering (see code for details)

    Returns:
        array of dithered samples (same shape as samples)
    """
    return samples + level*np.random.normal(0,1, samples.shape)
    

def lifter(mfcc, lifter=22):
    """
    Applies liftering to improve the relative range of MFCC coefficients.

       mfcc: NxM matrix where N is the number of frames and M the number of MFCC coefficients
       lifter: lifering coefficient

    Returns:
       NxM array with lifeterd coefficients
    """
    nframes, nceps = mfcc.shape
    cepwin = 1.0 + lifter/2.0 * np.sin(np.pi * np.arange(nceps) / lifter)
    return np.multiply(mfcc, np.tile(cepwin, nframes).reshape((nframes,nceps)))

def hz2mel(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log(f/700 +1)

def trfbank(fs, nfft, lowfreq=133.33, linsc=200/3., logsc=1.0711703, nlinfilt=13, nlogfilt=27, equalareas=False):
    """Compute triangular filterbank for MFCC computation.

    Inputs:
    fs:         sampling frequency (rate)
    nfft:       length of the fft
    lowfreq:    frequency of the lowest filter
    linsc:      scale for the linear filters
    logsc:      scale for the logaritmic filters
    nlinfilt:   number of linear filters
    nlogfilt:   number of log filters

    Outputs:
    res:  array with shape [N, nfft], with filter amplitudes for each column.
            (N=nlinfilt+nlogfilt)
    From scikits.talkbox"""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    if equalareas:
        heights = np.ones(nfilt)
    else:
        heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank

def dtw(locd):
  H, K = locd.shape
  accd = np.zeros((H, K))
  for h in range(0, H):
    for k in range(0, K):
      accd[h, k] = locd[h, k] + min(accd[h - 1, k], accd[h - 1, k - 1], accd[h, k - 1])
  return accd[-1,-1]

def locd(v1, v2):
  loc_d = np.zeros((len(v1),len(v2)))
  for x in range(len(v1)):
    for j in range(len(v2)):
      loc_d[x, j] = distance.euclidean(v1[x], v2[j])
  return loc_d


#Function to plot colormesh
def plot_colormesh(input):
  plt.pcolormesh(input.T)
  plt.show()