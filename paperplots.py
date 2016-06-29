import numpy as np
from matplotlib.pyplot import *
from astropy.io import fits
from sklearn import linear_model
from scipy.signal import lombscargle
from sklearn.utils import ConvergenceWarning
from methods import findPeaks, createCustomMatrix
import itertools, sys, pickle, datetime, os, warnings, ipdb, bisect, matplotlib


def plotSomething():
    # matplotlib.rcParams['font.weight'] = 'light'
    # matplotlib.rc('text', usetex=True)
    # rc('text', usetex=True)
    # rc('font', family='sans-serif')

    matplotlib.rcParams['font.sans-serif'] = ['Arial']

    with open('data/out_lasso_path3/amp1.00/data.pkl', 'rb') as f:
        data = pickle.load(f)
    imshow(data['deltadeltaP'], interpolation='nearest', aspect='auto', vmin=0, vmax=10,
        extent=[data['grid'][0,0,1], data['grid'][-1,-1,1], data['grid'][-1,-1,0], data['grid'][0,0,0]])
    yticks(range(-9, 10, 3))
    title('$\\Delta(\\Delta P_\\mathrm{{input}}, \\Delta P_\\mathrm{{output}})$')
    xlabel('SNR')
    ylabel('Input $\\Delta P$  [days]')
    colorbar()
    savefig('Paper/figures/test.pdf', bbox_inches='tight')
    close()
    # show()


if __name__ == '__main__':
    # matplotlib.rcParams
    plotSomething()
