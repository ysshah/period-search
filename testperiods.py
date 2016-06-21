import numpy as np
from matplotlib.pyplot import *
from sklearn import linear_model
from sklearn.utils import ConvergenceWarning
from methods import findPeaks, createCustomMatrix
import itertools, sys, pickle, datetime, os, warnings, ipdb


DIRNAME = 'out2'


def init():
    """Initialize parameters and variables."""
    time_i, time_f, dt = 0.0, 100.0, 0.02043359821692
    search_low, search_high, search_num = 10.0, 30.0, 1000
    delta_p_range = np.linspace(-9, 9, 40)
    SNR_range = np.linspace(1, 10, 40)
    p = 20.0

    noise_std_range = 1.0 / np.sqrt(SNR_range)
    time = np.arange(time_i, time_f, dt)

    search_periods = np.linspace(search_low, search_high, search_num)

    A = createCustomMatrix(time, search_periods)

    m = itertools.product(delta_p_range, SNR_range)
    variables = [a[0] + (a[1],) for a in zip(itertools.product(
        delta_p_range, SNR_range), range(delta_p_range.size*SNR_range.size))]

    grid = np.array(list(m)).reshape((delta_p_range.size, SNR_range.size, 2))

    return A, time, p, delta_p_range, SNR_range, variables, grid, search_periods


def createAlphaDirs(alpha):
    """Create necessary directories and filenames."""
    alphadir = os.path.join(DIRNAME, 'alpha_{:.1e}'.format(alpha))
    plotdir = os.path.join(alphadir, 'plots')
    rfile = os.path.join(alphadir, 'results.txt')
    figfile = os.path.join(alphadir, 'plot.png')
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    return alphadir, plotdir, rfile, figfile


def createArrays(N_dp, N_snr, search_num):
    """Initialize empty arrays for filling later."""
    diffs = np.empty((N_dp, N_snr))
    diffs.fill(np.nan)
    powers = np.empty((N_dp, N_snr, search_num))
    return diffs, powers


def plotImage(diffs, delta_p_range, SNR_range, alpha):
    """Create the main difference image plot."""
    fig = figure()
    img = imshow(diffs, aspect='auto', interpolation='nearest', extent=[
        SNR_range[0], SNR_range[-1], delta_p_range[-1], delta_p_range[0]])
    colorbar()
    title('$\\Delta(\\Delta P_\\mathrm{{input}}, \\Delta P_\\mathrm{{output}})$; '
          'alpha = {:.1e}'.format(alpha))
    xlabel('SNR ($1/\\sigma_\\mathrm{{noise}}^2$)')
    ylabel('Input $\\Delta$ period')
    return fig, img


def LassoDiffWrapper(args):
    return LassoDiff(*args)


def LassoDiff(clf, A, time, p, delta_p, SNR, i=None):
    flux = np.sin(2*np.pi/p * time) + np.sin(2*np.pi/(p + delta_p) * time)
    flux += np.random.normal(0, flux.std()/np.sqrt(SNR), flux.size)

    convWarning = False
    with warnings.catch_warnings(record=True) as w:
        clf.fit(A, flux)
        # ipdb.set_trace()
        if len(w) == 1 and type(w[0].message) is ConvergenceWarning:
            convWarning = True

    power = clf.coef_[:A.shape[1]//2]**2 + clf.coef_[A.shape[1]//2:A.shape[1]//2 * 2]**2

    peaks = findPeaks(power)
    if peaks.size > 1:
        d2P_dp2 = np.gradient(np.gradient(power))
        p1_i, p2_i = peaks[-4:][d2P_dp2[peaks[-4:]].argsort()][:2]
    else:
        p1_i, p2_i = power.argsort()[-2:]

    return i, power, p1_i, p2_i, convWarning


def recordProgress(start, n, N_pix, rfile):
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


def update(i, diffs, diff, powers, power, fig, img):
    """Update the data arrays and difference image plot."""
    r, c = np.unravel_index(i, diffs.shape)
    diffs[r,c] = diff
    powers[r,c] = power
    img.set_data(diffs)
    img.set_clim(np.nanmin(diffs), np.nanmax(diffs))
    fig.canvas.draw()
    fig.savefig(figfile, dpi=200)


def plotCoefficients(variables, i, search_periods, p1_i, p2_i, diff, power, alpha, plotdir, convWarning):
    """Plot the coefficients."""
    dp, SNR = variables[i][:2]
    p1_input, p2_input = 20, 20 + dp
    p1_output, p2_output = search_periods[[p1_i, p2_i]]
    f, ax = subplots()
    ax.set_title('$P_1 = {:.2f}, P_2 = {:.2f},$ SNR = {:.2f}, alpha = {:.1e}'.format(p1_input, p2_input, SNR, alpha))
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('Lasso Power')
    ax.axvline(p1_input, 0, 1, c='b')
    ax.axvline(p2_input, 0, 1, c='b', label='Input periods')
    color = 'r' if convWarning else 'g'
    label = 'Convergence Warning' if convWarning else 'Lasso converged'
    loc = 'upper right' if dp < 0 else 'upper left'
    ax.plot(search_periods, power, color+'o-', ms=4, label=label)
    ax.plot(search_periods[[p1_i, p2_i]], power[[p1_i, p2_i]], 'o',
        mec='m', mew=2, mfc='None', ms=10, label='Detected Periods')
    ax.margins(0.05, 0.05)
    ax.legend(fontsize=10, loc=loc)
    # if dp < 0:
    #     kwargs = {'xy':(0.99, 0.8), 'ha':'right'}
    # else:
    #     kwargs = {'xy':(0.01, 0.8), 'ha':'left'}
    # ax.annotate('$\\Delta P_\\mathrm{{in}} = {0:.2f}$\n'
    #     '$\\Delta P_\\mathrm{{out}} = {1:.2f}$\n'
    #     '$\\Delta(\\Delta P_\\mathrm{{in}}, \\Delta P_\\mathrm{{out}}) = {2:.2f}$'.format(abs(dp), abs(p1_output - p2_output), diff),
    #     xycoords='axes fraction', va='top', **kwargs)
    f.savefig(os.path.join(plotdir, '{:d}.png'.format(i)), dpi=200)
    close(f)


def alphaFinal(diffs, powers, grid, alphadir, rfile, start):
    """Save data and output last line to progress file."""
    data = {
        'diffs': diffs,
        'powers': powers,
        'grid': grid
    }
    with open(os.path.join(alphadir, 'data.pkl'), 'wb') as dataFile:
        pickle.dump(data, dataFile)
    with open(rfile, 'a') as f:
        f.write('Saved data file. Took {} to complete.\n'.format(
            str(datetime.datetime.now() - start).split('.')[0]))


if __name__ == '__main__':

    A, time, p, delta_p_range, SNR_range, variables, grid, search_periods = init()
    N_pix = delta_p_range.size * SNR_range.size

    a0 = 0.00001
    alpha_vars = (a0*2, a0/4, a0/2, a0, a0*4)
    for alpha in alpha_vars:
        alphadir, plotdir, rfile, figfile = createAlphaDirs(alpha)
        diffs, powers = createArrays(delta_p_range.size, SNR_range.size, search_periods.size)
        fig, img = plotImage(diffs, delta_p_range, SNR_range, alpha)
        clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
        args = [(clf, A, time, p,) + a for a in variables]
        start = datetime.datetime.now()

        for n, (i, power, p1_i, p2_i, convWarning) in enumerate(map(LassoDiffWrapper, args), 1):
            recordProgress(start, n, N_pix, rfile)
            p1, p2 = search_periods[[p1_i, p2_i]]
            diff = abs(abs(p1 - p2) - abs(variables[i][0]))
            update(i, diffs, diff, powers, power, fig, img)
            plotCoefficients(variables, i, search_periods, p1_i, p2_i, diff, power, alpha, plotdir, convWarning)

        close(fig)

        alphaFinal(diffs, powers, grid, alphadir, rfile, start)
