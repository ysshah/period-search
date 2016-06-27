import numpy as np
from matplotlib.pyplot import *
from astropy.io import fits
from sklearn import linear_model
from scipy.signal import lombscargle
from sklearn.utils import ConvergenceWarning
from methods import findPeaks, createCustomMatrix
import itertools, sys, pickle, datetime, os, warnings, ipdb, bisect


################### INPUT VARIABLES ###################
DIRNAME = 'out_lasso_path2'
time_i, time_f, dt = 0.0, 100.0, 0.02043359821692
search_low, search_high, search_num = 10.0, 30.0, 1000
delta_p_range = np.linspace(-9, 9, 40)
SNR_range = np.linspace(1, 10, 40)
center_period = 20.0
#######################################################

# Calculated variables
time = np.arange(time_i, time_f, dt)
kep = ~np.isnan(fits.open('001869783/kplr001869783-2009350155506_llc.fits')[1].data['PDCSAP_FLUX'])
kep = np.append(kep, np.array([True]*(time.size - kep.size)))

N_pix = delta_p_range.size * SNR_range.size
noise_std_range = 1.0 / np.sqrt(SNR_range)
search_periods = np.linspace(search_low, search_high, search_num)
m = itertools.product(delta_p_range, SNR_range)
variables = list(itertools.product(delta_p_range, SNR_range))
grid = np.array(list(m)).reshape((delta_p_range.size, SNR_range.size, 2))
A = createCustomMatrix(time[kep], search_periods)


def lassoPathWrapper(args):
    return lassoPath(*args)


def lassoPath(delta_p, SNR, a1, a2):
    flux = a1*np.sin(2*np.pi/center_period * time) + a2*np.sin(2*np.pi/(center_period + delta_p) * time)
    flux += np.random.normal(0, flux.std()/np.sqrt(SNR), flux.size)

    convWarning = False
    with warnings.catch_warnings(record=True) as w:
        clf = linear_model.lasso_path(A, flux[kep])
        if len(w) == 1 and type(w[0].message) is ConvergenceWarning:
            convWarning = True

    coeffs = clf[1][:,-1]
    power = np.sqrt(coeffs[:A.shape[1]//2]**2 + coeffs[A.shape[1]//2:A.shape[1]//2 * 2]**2)

    peaks = findPeaks(power)
    if peaks.size > 1:
        p1_i, p2_i = np.sort(peaks[-2:])[::int(np.sign(delta_p))]
    else:
        p1_i, p2_i = np.sort(power.argsort()[-2:])[::int(np.sign(delta_p))]

    f, ax1 = subplots()
    ax1.set_title('$P_1$ = {:.2f}, $A_1$ = {:.2f}, $P_2$ = {:.2f}, $A_2$ = {:.2f}, SNR = {:.2f}'.format(
        center_period, a1, center_period+delta_p, a2, SNR))

    l = []

    ax1.set_xlabel('Period [days]')
    ax1.set_ylabel('Lasso Power')
    ax1.axvline(center_period, 0, 1, c='r')
    l.append(ax1.axvline(center_period + delta_p, 0, 1, c='r', label='Input periods'))
    color = 'r' if convWarning else 'g'
    label = 'Convergence Warning' if convWarning else 'Lasso converged'
    loc = 'upper right' if delta_p < 0 else 'upper left'
    l.append(ax1.plot(search_periods, power, color+'o-', ms=4, label=label)[0])
    l.append(ax1.plot(search_periods[[p1_i, p2_i]], power[[p1_i, p2_i]], 'o',
        mec='m', mew=2, mfc='None', ms=10, label='Detected Periods')[0])

    z = np.where(power == 0)[0]
    colors = ('yellow', 'cyan')
    areas = []
    for j, p_i in enumerate((p1_i, p2_i)):
        i = bisect.bisect_left(z, p_i)
        maxIndex = power.size-1 if i >= len(z) else min(power.size-1, z[i]+1)
        minIndex = 0 if i == 0 else max(0, z[i-1])
        areas.append(np.trapz(power[minIndex:maxIndex], search_periods[minIndex:maxIndex]))
        l.append(ax1.fill_between(search_periods[minIndex:maxIndex], 0, power[minIndex:maxIndex],
            facecolor=colors[j], label='$P_{:d}$ Area = {:.2f}'.format(j+1, areas[j])))

    ax2 = ax1.twinx()
    pgram = lombscargle(time[kep], flux[kep], 2*np.pi/search_periods)
    l.append(ax2.plot(search_periods, pgram * 2/(flux.size * np.nanstd(flux)**2), 'b', label='Periodogram')[0])
    ax2.set_ylabel('Periodogram Power')

    ax1.margins(0.05, 0.05)
    ax2.margins(0.05, 0.05)
    ax1.legend(l, [a.get_label() for a in l], loc=loc, fontsize=10)

    dp1 = abs(center_period - search_periods[p1_i])
    dp2 = abs(center_period+delta_p - search_periods[p2_i])
    if convWarning:
        da1, da2 = np.nan, np.nan
    else:
        da1, da2 = abs(a1 - areas[0]), abs(a2 - areas[1])
    ddp = abs(abs(delta_p) - abs(search_periods[p1_i] - search_periods[p2_i]))

    return f, ddp, dp1, dp2, da1, da2, power


def createDirs(amplitude):
    """Create necessary directories and filenames."""
    ampdir = os.path.join(DIRNAME, 'amp{:.2f}'.format(amplitude))
    plotdir = os.path.join(ampdir, 'plots')
    rfile = os.path.join(ampdir, 'results.txt')
    figfiles = [os.path.join(ampdir, f) for f in [
        'deltadeltaP',
        'deltaP1',
        'deltaP2',
        'deltaA1',
        'deltaA2'
    ]]
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    pickledir = os.path.join('pickle_files', ampdir)
    if not os.path.isdir(pickledir):
        os.makedirs(pickledir)
    return ampdir, plotdir, rfile, figfiles


def createArrays():
    """Initialize empty arrays for filling later."""
    diff_arrays = np.empty((5, delta_p_range.size, SNR_range.size))
    diff_arrays.fill(np.nan)
    powers = np.empty((delta_p_range.size, SNR_range.size, search_num))
    return diff_arrays, powers


def createImages(diff_arrays, a1, a2):
    """Create difference image plots."""
    figs, imgs = [], []
    titles = [
        '$\\Delta(\\Delta P_\\mathrm{{input}}, \\Delta P_\\mathrm{{output}})$',
        '$\\Delta P_1$',
        '$\\Delta P_2$',
        '$\\Delta A_1$',
        '$\\Delta A_2$'
    ]
    for i in range(diff_arrays.shape[0]):
        figs.append(figure())
        imgs.append(imshow(diff_arrays[i], aspect='auto', interpolation='nearest', extent=[
            SNR_range[0], SNR_range[-1], delta_p_range[-1], delta_p_range[0]]))
        colorbar()
        title(titles[i]+'; $A_1$ = {:.2f}, $A_2$ = {:.2f}'.format(a1, a2))
        xlabel('SNR ($1/\\sigma_\\mathrm{{noise}}^2$)')
        ylabel('Input $\\Delta$ period')
    return figs, imgs   


def updateArrays(i, diff_arrays, ddp, dp1, dp2, da1, da2, power):
    """Update the data arrays and difference image plot."""
    r, c = np.unravel_index(i, diff_arrays[0].shape)
    diff_arrays[:,r,c] = ddp, dp1, dp2, da1, da2
    powers[r,c] = power


def updateImages(diff_arrays, figs, imgs, figfiles):
    """Update each image plot with new data."""
    for i in range(diff_arrays.shape[0]):
        imgs[i].set_data(diff_arrays[i])
        imgs[i].set_clim(np.nanmin(diff_arrays[i]), np.nanmax(diff_arrays[i]))
        figs[i].canvas.draw()
        figs[i].savefig(figfiles[i], dpi=200)


def recordProgress(start, n, rfile):
    stop = datetime.datetime.now()
    avg = (stop - start).total_seconds() / n
    remain = datetime.timedelta(seconds = avg * (N_pix - n))
    eta = stop + remain
    progress = '{:d}/{:d} pixels completed. Avg {:.2f} sec/pix, {:s} remaining, ETA {:s}    '.format(n, N_pix, avg,
        str(remain).split('.')[0], eta.strftime('%x %I:%M:%S %p'))
    with open(rfile, 'w') as f:
        f.write(progress+'\n')
    sys.stdout.write('\r'+progress)
    sys.stdout.flush() 


def subFinal(diff_arrays, powers, ampdir, rfile, start):
    """Save data and output last line to progress file."""
    ddp_array, dp1_array, dp2_array, da1_array, da2_array = diff_arrays
    data = {
        'deltadeltaP': ddp_array,
        'deltaP1': dp1_array,
        'deltaP2': dp2_array,
        'deltaA1': da1_array,
        'deltaA2': da2_array,
        'powers': powers,
        'grid': grid
    }
    with open(os.path.join('pickle_files', ampdir, 'data.pkl'), 'wb') as dataFile:
        pickle.dump(data, dataFile)
    with open(rfile, 'a') as f:
        f.write('Saved data file. Took {} to complete.\n'.format(
            str(datetime.datetime.now() - start).split('.')[0]))


if __name__ == '__main__':

    a1 = 1.00
    amp_vars = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
    # amp_vars = [1.00]
    for a2 in amp_vars[::-1]:
        ampdir, plotdir, rfile, figfiles = createDirs(a2)
        diff_arrays, powers = createArrays()
        figs, imgs = createImages(diff_arrays, a1, a2)
        args = [a + (a1, a2) for a in variables]
        start = datetime.datetime.now()

        for i, (f, ddp, dp1, dp2, da1, da2, power) in enumerate(map(lassoPathWrapper, args)):
            f.savefig(os.path.join(plotdir, '{:d}.png'.format(i)), dpi=200)
            close(f)
            updateArrays(i, diff_arrays, ddp, dp1, dp2, da1, da2, power)
            updateImages(diff_arrays, figs, imgs, figfiles)
            recordProgress(start, i+1, rfile)

        close('all')

        subFinal(diff_arrays, powers, ampdir, rfile, start)
