import numpy as np
import matplotlib.pyplot as plt
import matplotlib, warnings
from astropy.io import fits
from scipy.signal import lombscargle

def get_phase_offset(x, y, omega):
    """Get the phase offset of a fit to X, Y with frequency OMEGA."""
    return np.arctan2(np.nansum(y * np.sin(omega * x)),
                      np.nansum(y * np.cos(omega * x)))

# Open .fits file for KIC 1869783, grab time and flux
hdulist = fits.open('001995351/kplr001995351-2009350155506_llc.fits')
data = hdulist[1].data
time = data['TIME'].byteswap().newbyteorder().astype('float64')
flux = data['PDCSAP_FLUX'].byteswap().newbyteorder().astype('float64')

# Normalize flux and remove offset
flux = (flux - np.nanmean(flux)) / np.nanmean(flux) * 100

# Bin the data at 2 hour bins => 4 data points in long cadence
binwidth = 4
with warnings.catch_warnings(): # Suppress 'mean of empty slice' warnings for all NaN bins
    warnings.simplefilter('ignore', category=RuntimeWarning)
    binned_time = np.nanmean(time[:time.size//binwidth * binwidth].reshape(-1, binwidth), axis=1)
    binned_flux = np.nanmean(flux[:flux.size//binwidth * binwidth].reshape(-1, binwidth), axis=1)

# Period search space
periods = np.linspace(0.5, 50.0, 10000)
ang_freqs = 2*np.pi / periods

# Plotting and array initializations
ffig, ax = plt.subplots(figsize=(16, 3))
ax.plot(binned_time, binned_flux, 'k', lw=2)
depth = 5 # Number of periodograms
fig, axarr = plt.subplots(depth, 1, sharex=True, figsize=(16, depth*3))
total_fit = np.zeros_like(binned_flux)

# Loop to create 5 periodograms
for i in range(depth):
    # Remove NaN data from flux before supplying to function
    t = binned_time[~np.isnan(binned_flux)]
    f = (binned_flux - total_fit)[~np.isnan(binned_flux)]
    pgram = lombscargle(t, f, ang_freqs)

    # Normalize periodogram power
    power = pgram * 2 / (f.size * f.std()**2)

    # Get fit parameters using maximum power and its period
    j = power.argmax()
    period = periods[j]
    # Not sure how exactly to calculate amplitude below
    amplitude = np.sqrt(2 * pgram.max() * np.nanstd(binned_flux)**2)
    phi = get_phase_offset(binned_time, binned_flux - total_fit, ang_freqs[j])
    fit = amplitude * np.cos(binned_time * 2*np.pi/period - phi)

    # Add fit to total fit
    total_fit += fit

    # Plot periodogram
    axarr[i].plot(periods, power, 'k')
    axarr[i].vlines(period, 0, power.max(), colors='r')

# Plot total fit over original binned lightcurve
ax.set_ylabel('Rel. flux (%)')
ax.set_title('KIC: 1995351')
ax.plot(binned_time, total_fit, 'r', lw=2)
ax.margins(x=0.05)
ax.set_xticks(range(260, 360, 20))
ax.set_ylim(-1.5, 1.5)
ax.set_yticks(np.arange(-1.5, 2, 0.5))

# Label the periodogram plots
axarr[0].set_title('BJD - 2454833')
axarr[0].set_yticks(np.arange(0, 1, 0.2))
for i in range(1, depth):
    axarr[i].set_ylim(0, 0.3)
    axarr[i].set_yticks(np.arange(0, 0.3, 0.05))
axarr[2].set_ylabel('Norm. Power')
axarr[-1].set_xlabel('Period [d]')
axarr[-1].set_xscale('log')
axarr[-1].set_xlim(periods[0], periods[-1])
axarr[-1].set_xticks([1, 10])
axarr[-1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig.subplots_adjust(hspace=0)
plt.show()
