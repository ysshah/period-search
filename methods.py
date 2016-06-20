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


def prewhitenExample():
    # time, flux, dt = getKeplerData('../001995351/kplr001995351-2009350155506_llc.fits')
    time, flux, dt = getKeplerData('../001869783/kplr001869783-2009350155506_llc.fits')
    time, flux = removeOutliers(time, flux)
    time, flux = binData(time, flux, 4)
    # flux -= np.nanmean(flux)
    # flux /= np.nanstd(flux)
    flux = (flux - np.nanmean(flux)) / np.nanmean(flux) * 100

    # search_low = 0.5
    search_low = 7.5
    # search_high = 14.0
    search_high = 110.0
    search_num = 10000
    depth = 5


    pfig, paxarr = subplots(depth, 1, sharex=True, figsize=(28,16))
    # ffig, faxarr = subplots(depth, 1, sharex=True)
    total_fit = np.zeros_like(flux)
    for i in range(depth):
        periods, power, phase = Lomb_Scargle_with_phase(time, flux - total_fit, search_low, search_high, search_num)
        j = power.argmax()
        per = periods[j]
        phi = phase[j]
        amp = np.sqrt(2*power.max())
        # amp = np.sqrt(power.max()) / np.std(np.cos(np.arange(time[0], time[-1]+0.5*dt, dt) * 2*np.pi/per - phi))
        fit = amp * np.cos(time * 2*np.pi/per - phi)

        # import ipdb
        # ipdb.set_trace()

        paxarr[i].set_ylabel('Periodogram %d'%(i+1))
        power /= np.nanstd(flux - total_fit)**2
        paxarr[i].plot(periods, power)
        paxarr[i].vlines(periods[j], 0, power.max(), colors='r')
        paxarr[i].tick_params('x', length=10, width=1, which='both')
        # faxarr[i].plot(time, flux - total_fit, label='Data for P%d calculation' %(i+1))
        # faxarr[i].plot(time, fit, label='Fit %d' %(i+1))

        total_fit += fit

    pfig.subplots_adjust(hspace=0)
    # ffig.subplots_adjust(hspace=0)
    paxarr[0].set_xscale('log')
    paxarr[0].set_xlim(search_low, search_high)
    # paxarr[-1].set_xticks([1, 10])
    paxarr[-1].set_xticks([10, 20, 30, 40, 50, 70, 100])
    paxarr[-1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    figure(figsize=(28, 8))
    plot(time, flux, label='Data')
    plot(time, total_fit, 'r', label='Fit')

    show()


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


def LassoDiff(delta_p, noise_std, i=None):
    p = 20.0
    dt = 0.02043359821692

    time = np.arange(0, 100, dt)
    input_periods = np.array([p, p + delta_p])
    frequencies = 2*np.pi / input_periods
    amplitudes = np.ones_like(input_periods)
    flux = np.dot(np.sin(frequencies * np.array([time]).T), amplitudes)
    flux /= flux.std()
    flux += np.random.normal(0, noise_std, len(flux))

    periods, power = customLasso(time, flux,
        alpha=0.0001, search_low=10.0, search_high=30.0, search_num=1000)

    peaks = findPeaks(power)
    if peaks.size > 1:
        p1, p2 = periods[peaks][-2:]
    else:
        p1, p2 = periods[power.argsort()][-2:]
    
    return i, abs(abs(p1 - p2) - abs(delta_p)), periods, power


def LassoDiffWrapper(args):
    return LassoDiff(*args)


def makeImage():
    dirname = 'out'
    plotdir = os.path.join(dirname, 'plots')
    rfile = os.path.join(dirname, 'results.txt')
    figfile = os.path.join(dirname, 'plot.png')
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    delta_p_range = np.linspace(-8, 8, 10)
    SNR_range = np.linspace(0.1, 1, 10)

    N_dp = delta_p_range.size
    N_snr = SNR_range.size
    N_pix = N_dp*N_snr

    print('%d delta_p x %d SNR = %d values' %(N_dp, N_snr, N_pix))

    diffs = np.empty((N_dp, N_snr))
    diffs.fill(np.nan)
    fig, img = plotDiffs(diffs, delta_p_range, SNR_range)

    p = Pool(8)
    deltap_noise_args = list(itertools.product(delta_p_range, 1.0 / np.sqrt(SNR_range)))
    args = [a[0] + (a[1],) for a in zip(deltap_noise_args, range(N_pix))]
    start = datetime.datetime.now()
    for n, (i, diff, periods, power) in enumerate(p.imap_unordered(LassoDiffWrapper, args), 1):
        progress = readable(start, n, N_pix)
        with open(rfile, 'w') as f:
            f.write(progress+'\n')
        sys.stdout.write('\r'+progress)
        sys.stdout.flush()

        diffs[np.unravel_index(i, diffs.shape)] = diff
        img.set_data(diffs)
        img.set_clim(np.nanmin(diffs), np.nanmax(diffs))
        fig.canvas.draw()
        fig.savefig(figfile, dpi=200)

        delta_p, noise_std = deltap_noise_args[i]
        SNR = 1.0/noise_std**2
        f, ax = subplots()
        ax.plot(periods, power, 'o-')
        ax.set_title('$\\Delta p =$ %f, SNR = %f' %(delta_p, SNR))
        ax.set_xlabel('Period [days]')
        ax.set_ylabel('Power')
        f.savefig(os.path.join(plotdir, '%d.png'%i), dpi=200)
        close(f)

    p.close()

    pickle.dump(diffs, open('diffs.pkl', 'wb'))
    with open(rfile, 'a') as f:
        f.write('Saved diffs file\n')


def findPeaks(x):
    """Find local maxima of time-ordered signal X."""
    peaks = np.r_[True, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:], True]
    peaks[[0, -1]] = False, False
    return np.where(peaks)[0][np.argsort(x[peaks])]


def plotDiffs(diffs, delta_p_range, SNR_range):
    fig = figure()
    img = imshow(diffs, aspect='auto', interpolation='nearest', extent=[
        SNR_range[0], SNR_range[-1], delta_p_range[0], delta_p_range[-1]])
    colorbar()
    xlabel('SNR ($1/\\sigma_\\mathrm{noise}^2$)')
    ylabel('Input $\\Delta$ period')
    return fig, img


def readable(start, n, N_pix):
    stop = datetime.datetime.now()
    avg = (stop - start).total_seconds() / n
    remain = datetime.timedelta(seconds = avg * (N_pix - n))
    eta = stop + remain
    return '%d/%d pixels completed. Avg %.2f sec/pix, %s remaining, ETA %s    ' %(
        n, N_pix, avg, str(remain).split('.')[0], eta.strftime('%x %I:%M:%S %p'))


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


def LassoDiff2(clf, A, p_indexes, SNR, i=None):
    x = np.zeros(A.shape[1])
    x[p_indexes] = 1.0 # np.sqrt(time.size/2)

    flux = np.dot(A, x)
    flux += np.random.normal(0, flux.std()/np.sqrt(SNR), flux.size)

    clf.fit(A, flux)
    diff = np.sum((clf.coef_ - x)**2)

    return i, diff, x, clf.coef_


def LassoDiff2Wrapper(args):
    return LassoDiff2(*args)


def makeImage2():
    dirname = 'out1'

    time_i, time_f, dt = 0.0, 100.0, 0.02043359821692
    search_low, search_high, search_num = 10.0, 30.0, 1000
    delta_p_range = np.linspace(-9, 9, 40)
    SNR_range = np.linspace(1, 10, 40)
    p = 20.0

    search_periods = np.linspace(search_low, search_high, search_num)
    p1 = np.argmin(np.abs(p - search_periods))
    p2 = np.argmin(np.abs(np.array([p + delta_p_range]).T - search_periods), axis=1)
    p_indexes = np.array(list(zip(itertools.repeat(p1), p2)))

    A = createCustomMatrix(np.arange(time_i, time_f, dt), search_periods)

    N_dp = delta_p_range.size
    N_snr = SNR_range.size
    N_pix = N_dp*N_snr
    variables = [a[0] + (a[1],) for a in zip(itertools.product(p_indexes, SNR_range), range(N_pix))]

    s = search_periods[p_indexes]
    dp = s[:,1] - s[:,0]
    grid = np.array(list(itertools.product(dp, SNR_range))).reshape((N_dp, N_snr, 2))

    a0 = 0.00001
    for alpha in (a0*2, a0/4, a0/2, a0, a0*4):

        alphadir = os.path.join(dirname, 'alpha_%.1e' %alpha)
        plotdir = os.path.join(alphadir, 'plots')
        rfile = os.path.join(alphadir, 'results.txt')
        figfile = os.path.join(alphadir, 'plot.png')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)

        diffs = np.empty((N_dp, N_snr))
        x_input = np.empty((N_dp, N_snr, A.shape[1]))
        x_output = np.empty((N_dp, N_snr, A.shape[1]))
        diffs.fill(np.nan)
        fig = figure()
        img = imshow(diffs, aspect='auto', interpolation='nearest', extent=[
            SNR_range[0], SNR_range[-1], delta_p_range[-1], delta_p_range[0]])
        colorbar()
        title('Residuals $\\sum (x_\\mathrm{input} - x_\\mathrm{fit})^2$; alpha = %.1e'%alpha)
        xlabel('SNR ($1/\\sigma_\\mathrm{noise}^2$)')
        ylabel('Input $\\Delta$ period')

        clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
        args = [(clf, A,) + a for a in variables]
        start = datetime.datetime.now()
        for n, (i, diff, x, coeffs) in enumerate(map(LassoDiff2Wrapper, args), 1):
            progress = readable(start, n, N_pix)
            with open(rfile, 'w') as f:
                f.write(progress+'\n')
            sys.stdout.write('\r'+progress)
            sys.stdout.flush()

            diffs[np.unravel_index(i, diffs.shape)] = diff
            x_input[np.unravel_index(i, diffs.shape)] = x
            x_output[np.unravel_index(i, diffs.shape)] = coeffs
            img.set_data(diffs)
            img.set_clim(np.nanmin(diffs), np.nanmax(diffs))
            fig.canvas.draw()
            fig.savefig(figfile, dpi=200)

            f, ax = subplots()
            p_i, SNR = args[i][2:4]
            p1, p2 = search_periods[p_i]
            ax.set_title('$P_1 = %f, P_2 = %f, \\Delta P = %f,$ SNR = %f' %(p1, p2, p2 - p1, SNR))
            ax.set_xlabel('Lasso Coefficient Number')
            ax.set_ylabel('Lasso Power')
            ax.plot(x, 'o-', label='Input $x$')
            ax.plot(coeffs, 'o-', label='Output $x$')
            ax.legend()
            ax.margins(0.05, 0.05)
            f.savefig(os.path.join(plotdir, '%d.png'%i), dpi=200)
            close(f)
        close(fig)

        obj = {
            'diffs': diffs,
            'x_input': x_input,
            'x_output': x_output,
            'grid': grid
        }

        pickle.dump(obj, open(os.path.join(alphadir, 'data.pkl'), 'wb'))
        with open(rfile, 'a') as f:
            f.write('Saved data file. Took %s to complete.\n' %(
                str(datetime.datetime.now() - start).split('.')[0]))


if __name__ == '__main__':

    dirname = 'out1'

    time_i, time_f, dt = 0.0, 100.0, 0.02043359821692
    search_low, search_high, search_num = 10.0, 30.0, 1000
    delta_p_range = np.linspace(-9, 9, 40)
    SNR_range = np.linspace(1, 10, 40)
    p = 20.0

    search_periods = np.linspace(search_low, search_high, search_num)
    p1 = np.argmin(np.abs(p - search_periods))
    p2 = np.argmin(np.abs(np.array([p + delta_p_range]).T - search_periods), axis=1)
    p_indexes = np.array(list(zip(itertools.repeat(p1), p2)))

    A = createCustomMatrix(np.arange(time_i, time_f, dt), search_periods)

    N_dp = delta_p_range.size
    N_snr = SNR_range.size
    N_pix = N_dp*N_snr
    variables = [a[0] + (a[1],) for a in zip(itertools.product(p_indexes, SNR_range), range(N_pix))]

    s = search_periods[p_indexes]
    dp = s[:,1] - s[:,0]
    grid = np.array(list(itertools.product(dp, SNR_range))).reshape((N_dp, N_snr, 2))

    a0 = 0.00001
    alpha_vars = (a0*2, a0/4, a0/2, a0, a0*4)
    for alpha in alpha_vars[:1]:

        alphadir = os.path.join(dirname, 'alpha_{:.1e}'.format(alpha))
        plotdir = os.path.join(alphadir, 'plots')
        rfile = os.path.join(alphadir, 'results.txt')
        figfile = os.path.join(alphadir, 'plot.png')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)

        diffs = np.empty((N_dp, N_snr))
        x_input = np.empty((N_dp, N_snr, A.shape[1]))
        x_output = np.empty((N_dp, N_snr, A.shape[1]))
        diffs.fill(np.nan)
        fig = figure()
        img = imshow(diffs, aspect='auto', interpolation='nearest', extent=[
            SNR_range[0], SNR_range[-1], delta_p_range[-1], delta_p_range[0]])
        colorbar()
        title('Residuals $\\sum (x_\\mathrm{{input}} - x_\\mathrm{{fit}})^2$; alpha = {:.1e}'.format(alpha))
        xlabel('SNR ($1/\\sigma_\\mathrm{{noise}}^2$)')
        ylabel('Input $\\Delta$ period')

        clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
        args = [(clf, A,) + a for a in variables]
        start = datetime.datetime.now()
        for n, (i, diff, x, coeffs) in enumerate(map(LassoDiff2Wrapper, args[:1]), 1):
            progress = readable(start, n, N_pix)
            with open(rfile, 'w') as f:
                f.write(progress+'\n')
            sys.stdout.write('\r'+progress)
            sys.stdout.flush()

            r, c = np.unravel_index(i, diffs.shape)
            diffs[r,c] = diff
            x_input[r,c] = x
            x_output[r,c] = coeffs
            img.set_data(diffs)
            img.set_clim(np.nanmin(diffs), np.nanmax(diffs))
            fig.canvas.draw()
            fig.savefig(figfile, dpi=200)

            f, ax1 = subplots()
            ax2 = ax1.twinx()
            p_i, SNR = args[i][2:4]
            p1, p2 = search_periods[p_i]
            ax1.set_title('$P_1 = {:.2f}, P_2 = {:.2f}, \\Delta P = {:.2f},$ SNR = {:.2f}'.format(p1, p2, p2 - p1, SNR))
            ax1.set_xlabel('Lasso Coefficient Number')
            ax1.set_ylabel('Lasso Power')
            l1, = ax1.plot(x, 'o-', 'b', label='Input $x$')
            l2, = ax2.plot(coeffs, 'o-', 'g', label='Output $x$')
            ax1.legend([l1, l2])
            ax1.margins(0.05, 0.05)
            f.savefig(os.path.join(plotdir, '{:d}.png'.format(i)), dpi=200)
            close(f)
        close(fig)

        obj = {
            'diffs': diffs,
            'x_input': x_input,
            'x_output': x_output,
            'grid': grid
        }

        pickle.dump(obj, open(os.path.join(alphadir, 'data.pkl'), 'wb'))
        with open(rfile, 'a') as f:
            f.write('Saved data file. Took {} to complete.\n'.format(
                str(datetime.datetime.now() - start).split('.')[0]))
