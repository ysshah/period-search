from methods import *

def leastSQ():
    from scipy.optimize import leastsq

    _, diff, periods, power = LassoDiff(-0.88888, 0.01)

    def double_gaussian(x, params):
        (c1, mu1, c2, mu2) = params
        sigma = 0.1
        return c1 * np.exp(-(x - mu1)**2.0 / (2.0 * sigma**2.0)) \
              + c2 * np.exp(-(x - mu2)**2.0 / (2.0 * sigma**2.0))

    def double_gaussian_fit(params):
        fit = double_gaussian(periods, params)
        return (fit - y_proc)

    y_proc = power.copy()

    init_params = [1750, 15, 500, 25]

    fit = leastsq(double_gaussian_fit, init_params)
    print('c1 = {}'.format(fit[0][0]))
    print('m1 = {}'.format(fit[0][1]))
    print('c2 = {}'.format(fit[0][2]))
    print('m2 = {}'.format(fit[0][3]))
    plot(periods, power, 'bo-')
    # plot(periods, double_gaussian(periods, init_params), 'r')
    # figure()
    x = np.linspace(periods[0], periods[-1], 10000)
    plot(x, double_gaussian(x, fit[0]), 'r-')
    show()


def gaussian_double_fit():
    from scipy.optimize import basinhopping

    _, diff, periods, power = LassoDiff(-0.88888, 0.1)

    def double_gaussian(x, params):
        (c1, mu1, c2, mu2) = params
        sigma = 0.1
        return c1 * np.exp(-(x - mu1)**2.0 / (2.0 * sigma**2.0)) \
             + c2 * np.exp(-(x - mu2)**2.0 / (2.0 * sigma**2.0))

    def double_gaussian_fit(params):
        return sum((y_proc - double_gaussian(periods, params))**2)

    def accept_test(f_new, x_new, f_old, x_old):
        return bool(10 <= x_new[1] and x_new[1] <= 30 and 10 <= x_new[3] and x_new[3] <= 30)

    y_proc = power.copy()
    init_params = [1750, 20, 0, 25]

    fit = basinhopping(double_gaussian_fit, init_params, accept_test=accept_test)
    print(fit)
    print('c1 = {}'.format(fit['x'][0]))
    print('m1 = {}'.format(fit['x'][1]))
    print('c2 = {}'.format(fit['x'][2]))
    print('m2 = {}'.format(fit['x'][3]))
    plot(periods, power, 'bo-')
    x = np.linspace(periods[0], periods[-1], 10000)
    plot(x, double_gaussian(x, init_params), 'g-')
    plot(x, double_gaussian(x, fit['x']), 'r-')
    show()


if __name__ == '__main__':
    gaussian_double_fit()
