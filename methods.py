import numpy as np
from matplotlib.pyplot import *
from astropy.io import fits
from scipy.signal import lombscargle
from sklearn import linear_model
import time as pythonTime
from multiprocessing import Pool
import itertools, sys, pickle, datetime, os


def getKeplerData(filename):
    """Retrieve Kepler data from local folder."""
    # Open file
    hdulist = fits.open(filename)
    data = hdulist[1].data

    # Delta time
    dt = hdulist[1].header['TIMEDEL']

    # Convert big-endian to little-endian
    time = data['TIME'].byteswap().newbyteorder().astype('float64')
    flux = data['PDCSAP_FLUX'].byteswap().newbyteorder().astype('float64')

    # If time begins or ends with NaN, time range calculation will be off
    assert not (np.isnan(time[0]) or np.isnan(time[-1]))

    # Normalize flux and remove offset
    # flux = (flux - np.nanmean(flux)) / np.nanmean(flux) * 100

    return time, flux, dt


def medianDetrend(flux, binWidth):
    """Median detrend FLUX at a bin of BINDWIDTH points."""
    halfNumPoints = binWidth // 2
    medians = []
    for i in range(len(flux)):
        if i < halfNumPoints:
            medians.append(np.nanmedian(flux[:i+halfNumPoints+1]))
        elif i > len(flux) - halfNumPoints - 1:
            medians.append(np.nanmedian(flux[i-halfNumPoints:]))
        else:
            medians.append(np.nanmedian(flux[i-halfNumPoints : i+halfNumPoints+1]))
    return flux - medians


def removeOutliers(time, flux):
    f = medianDetrend(flux, 3)
    outlierIdx = f > 4*np.nanstd(f)
    return time[~outlierIdx], flux[~outlierIdx]


def binData(time, flux, binWidth):
    numBins = int(np.ceil(len(time)/float(binWidth)))
    tbin = np.zeros(numBins)
    fbin = np.zeros(numBins)
    for i in range(numBins):
        tbin[i] = np.nanmean(time[i*binWidth:(i+1)*binWidth])
        fbin[i] = np.nanmean(flux[i*binWidth:(i+1)*binWidth])
    return tbin, fbin


def genSignal(t_initial, t_final, t_num, periods, amplitudes=None,
    drop_prob=0.0, noise_std=0.0):
    """Generate a periodic signal from T_INITIAL to T_FINAL with T_NUM data
    points.
    """

    if amplitudes is not None:
        assert len(periods) == len(amplitudes), 'Must have amplitudes for every period.'
    else:
        amplitudes = np.ones_like(periods)
        # amplitudes = np.full(len(periods), 1.0/len(periods))

    time = np.linspace(t_initial, t_final, t_num)
    frequencies = 2*np.pi / np.array(periods)
    signal = np.dot(np.sin(frequencies * np.array([time]).T), amplitudes)

    signal /= signal.std()

    if noise_std:
        signal += np.random.normal(0, noise_std, len(signal))
    signal[np.random.rand(len(signal)) < drop_prob] = np.nan

    return time, signal


def Lomb_Scargle(time, flux, p_low, p_high, p_num):
    """Compute the Lomb-Scargle periodogram of timestream TIME and FLUX,
    searching P_NUM periods between P_LOW and P_HIGH.
    """
    # Remove NaN data
    t = time[~np.isnan(flux)]
    f = flux[~np.isnan(flux)]

    # Create periodogram
    periods = np.linspace(p_low, p_high, p_num)
    frequencies = 2*np.pi / periods
    pgram = lombscargle(t, f, frequencies)
    power = pgram * 2 / (len(f) * f.std()**2)
    
    return periods, power


def preWhitening(time, flux, p_low, p_high, p_num, depth=5):
    """Return best fit to FLUX using Lomb-Scargle periodograms with DEPTH
    number of pre-whitening steps.
    """
    periods = np.linspace(p_low, p_high, p_num)
    ang_freqs = 2*np.pi / periods

    f = flux.copy()
    total_fit = np.zeros_like(flux)
    detected_periods = np.zeros(depth)
    powers = np.zeros((depth, p_num))

    for d in range(depth):
        pgram = lombscargle(time[~np.isnan(flux)], f, ang_freqs)
        power = pgram * 2 / (len(f) * f.std()**2)
        powers[d,:] = power
        i = power.argmax()
        detected_periods[d] = periods[i]
        fit = power[i] * np.cos(time * 2*np.pi/periods[i] - getPhaseOffset(time[~np.isnan(flux)], f, ang_freqs[i]))
        total_fit += fit
        f -= fit
    return total_fit, np.sort(detected_periods), periods, powers


def FFT(time, flux):
    """Do a NumPy FFT of TIME and FLUX."""
    # Set NaN data to 0
    f = np.nan_to_num(flux)

    # Number of data points and range of time
    N = len(time)
    T = time[-1] - time[0]

    # Throw out the infinite period value
    power = (np.abs(np.fft.fft(f)) * 2.0//N)[1:N//2]
    frequencies = np.fft.fftfreq(N)[1:N//2]
    periods = (T / N) / frequencies

    return periods, power


def createMatrix(nrows, times, N, T, multiplier=1):
    """Create a basis matrix of sines and cosines."""
    A = np.zeros((nrows, N * multiplier))
    k0 = np.arange(0, N//2, 1./multiplier)
    k1 = np.arange(N//2, N, 1./multiplier)
    A[:,np.arange(0, N//2 * multiplier)] = np.cos(k0*times*2*np.pi/T)
    A[:,np.arange(N//2 * multiplier, N*multiplier)] = np.sin((k1-N//2)*times*2*np.pi/T)

    # Normalize the matrix
    A *= np.sqrt(2./N)
    A[:,0] /= np.sqrt(2.)

    return A


def Matrix(time, flux):
    """Create a basis matrix of sines and cosines, and return the powers of the
    coefficient vector.
    """
    # Set NaN data to 0
    f = np.nan_to_num(flux)

    # Number of data points and range of time
    N = len(time)
    T = time[-1] - time[0]

    # Time values for the matrix, as a column vector
    times = np.array([np.linspace(0, T, N)]).T

    # Create matrix, calculate coefficients, retrieve power spectrum
    A = createMatrix(N, times, N, T)
    coeffs = np.dot(A.T, f)
    power = 2.0/(N-1) * (coeffs[1:N//2]**2 + coeffs[N//2+1:N//2*2]**2)
    periods = T / np.arange(1, N//2)

    return periods, power


def Lasso(time, flux, alpha, multiplier=1):
    """Create a basis matrix with missing rows dependent on the data, and use
    the Lasso method to retrieve the coefficient vector.
    """
    assert type(multiplier) is int, 'Multiplier must be an integer.'

    # Get indices of valid flux values
    flux_idx = ~np.isnan(flux)

    # Number of data points and range of time
    N = len(time)
    T = time[-1] - time[0]

    # Time values for the matrix, as a column vector
    times = np.array([(time - time[0])[flux_idx]]).T

    # Create matrix, fit using Lasso, retrieve coefficients
    A = createMatrix(flux_idx.sum(), times, N, T, multiplier)
    clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    clf.fit(A, flux[flux_idx])
    coeffs = clf.coef_
    power = 2.0/(N-1) * (coeffs[1:N//2 * multiplier]**2 + coeffs[(N//2 * multiplier)+1:N//2*2 * multiplier]**2)
    periods = T / np.arange(1./multiplier, N//2, 1./multiplier)

    return periods, power


def createCustomMatrixWithDeltas(nrows, times, periods):
    p_num = len(periods)
    frequencies = 2*np.pi / periods
    A = np.zeros((nrows, p_num * 2 + nrows))
    A[:,:p_num] = np.cos(times * frequencies)
    A[:,p_num:2*p_num] = np.sin(times * frequencies)

    A[:,2*p_num:] = np.identity(nrows)

    # A *= np.sqrt(2./nrows)

    return A


def customLassoWithDeltas(time, flux, alpha, p_low, p_high, p_num):
    # Get indices of valid flux values
    flux_idx = ~np.isnan(flux)

    # Number of data points and range of time
    N = len(time)
    T = time[-1] - time[0]

    # Time values for the matrix, as a column vector
    times = np.array([(time - time[0])[flux_idx]]).T

    # Create matrix, fit using Lasso, retrieve coefficients
    periods = np.linspace(p_low, p_high, p_num)
    A = createCustomMatrixWithDeltas(flux_idx.sum(), times, periods)
    clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    clf.fit(A, flux[flux_idx])
    coeffs = clf.coef_
    power = coeffs[:p_num]**2 + coeffs[p_num:2*p_num]**2

    deltaPowers = coeffs[2*p_num:]

    return periods, power, deltaPowers


def testAll(time, signal, p_low, p_high, p_num, alpha, input_periods=None):
    start = pythonTime.time()
    LS_periods, LS_power = Lomb_Scargle(time, signal, p_low, p_high, p_num)
    endLS = pythonTime.time()
    print('Lomb-Scargle took', endLS - start, 'seconds')

    FFT_periods, FFT_power = FFT(time, signal)
    endFFT = pythonTime.time()
    print('FFT took', endFFT - endLS, 'seconds')

    Matrix_periods, Matrix_power = Matrix(time, signal)
    endMat = pythonTime.time()
    print('Matrix took', endMat - endFFT, 'seconds')

    Lasso_periods, Lasso_power = Lasso(time, signal, alpha)
    endLas = pythonTime.time()
    print('Lasso took', endLas - endMat, 'seconds')

    fig, ax = subplots(figsize=(16,4))
    ax.plot(time, signal)
    ax.set_title('Timestream')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Relative Flux [%]')
    fig.tight_layout()

    f, axarr = subplots(4, 1, figsize=(16,16))
    axarr[0].plot(LS_periods, LS_power, 'k', label='Power')
    axarr[0].set_title('Lomb-Scargle Periodogram')
    axarr[1].plot(FFT_periods, FFT_power, 'ko-', label='Power')
    axarr[1].set_title('Fourier Transform')
    axarr[2].plot(Matrix_periods, Matrix_power, 'ko-', label='Power')
    axarr[2].set_title('Basis Matrix Coefficient Power')
    axarr[3].plot(Lasso_periods, Lasso_power, 'ko-', label='Power')
    axarr[3].set_title('Lasso Power')
    axarr[3].set_xlabel('Period [days]')
    for ax in axarr[1:]:
        ax.set_xlim(p_low, p_high)
    if input_periods:
        for ax in axarr:
            for p in input_periods:
                ax.axvline(p, 0, 1, c='g', label='P = %f'%p)
            ax.legend()
    f.tight_layout()


def getPhaseOffset(x, y, omega):
    """Get the phase offset of a sinusoidal fit with frequency OMEGA """
    return np.arctan2(np.nansum(y * np.sin(omega * x)),
                      np.nansum(y * np.cos(omega * x)))


def createCustomMatrix(time, periods, nrows=None):
    time = np.array([time]).T
    frequencies = 2*np.pi / periods
    if nrows:
        A = np.zeros((nrows, periods.size * 2))
    else:
        A = np.zeros((time.size, periods.size * 2))
    A[:,:periods.size] = np.cos(time * frequencies)
    A[:,periods.size:] = np.sin(time * frequencies)

    A *= np.sqrt(2.0/time.size)

    return A


def customLasso(time, flux, alpha, search_low, search_high, search_num):
    # Get indices of valid flux values
    flux_idx = ~np.isnan(flux)

    # Number of data points and range of time
    N = len(time)
    T = time[-1] - time[0]

    # Time values for the matrix, as a column vector
    times = (time - time[0])[flux_idx]

    # Create matrix, fit using Lasso, retrieve coefficients
    periods = np.linspace(search_low, search_high, search_num)
    A = createCustomMatrix(times, periods, flux_idx.sum())
    clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    clf.fit(A, flux[flux_idx])
    coeffs = clf.coef_
    power = coeffs[:search_num]**2 + coeffs[search_num:]**2

    return periods, power


def findPeaks(x):
    """Find local maxima of time-ordered signal X."""
    peaks = np.r_[True, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:], True]
    peaks[[0, -1]] = False, False
    return np.where(peaks)[0][np.argsort(x[peaks])]


if __name__ == '__main__':
    pass
