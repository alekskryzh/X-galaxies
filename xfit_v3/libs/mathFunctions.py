#! /usr/bin/env python

import math
import numpy as np
from scipy.optimize import differential_evolution


def get_flux_smart_way(data, x, y):
    """ Function returns flux at the given coordinates with taking into account
    that they can have not integer values"""
    x, y = y, x  # Take inverse coordinate grid in a its file into account
    xMax, yMax = data.shape[0], data.shape[1]
    if (x >= xMax-1) or (y >= yMax-1) or (x < 1) or (y < 1):
        return 0
    fx, ix = math.modf(x)
    fy, iy = math.modf(y)
    ix = int(ix)
    iy = int(iy)
    flx = ((1.0-fx)*(1.0-fy)*data[ix, iy] + fx*(1.0-fy)*data[ix+1, iy] +
           fy*(1.0-fx)*data[ix, iy+1] + fx*fy*data[ix+1, iy+1])
    return flx


def fit_by_line(xs, ys, sigmas):
    # Rid off zero values in sigmas so we can devide by it
    sigmas[sigmas == 0.0] = np.min(sigmas[sigmas != 0])

    def objective(params):
        slope = params[0]
        intercept = params[1]
        line_predicted = slope * xs + intercept
        diff = (ys - line_predicted) ** 2.0
        fitness = np.average(diff, weights=1/sigmas)
        return fitness
    inds_x = np.argsort(xs)  # in case if points are not sorted
    delta_x = xs[inds_x[-1]] - xs[inds_x[0]]
    delta_y = ys[inds_x[-1]] - ys[inds_x[0]]
    slope_guess = delta_y / delta_x
    intercept_guess = ys[inds_x[0]]
    bounds = ((slope_guess/10, slope_guess*10), (intercept_guess/10, intercept_guess*10))
    fit_result = differential_evolution(func=objective, bounds=bounds, polish=True)
    return fit_result.x


def explaw(params, x):
    central_value = params[0]
    exp_scale = params[1]
    return central_value * np.exp(-x/exp_scale)


def fit_by_explaw(xs, ys, sigmas):
    """
    The function fits the exponential law into the data. The objective
    function is like f0 * exp(-r/h), where f0 -- is the value of the function
    in the coordinate origin, and h -- exponential scale length.
    """
    # Rid off zero values in sigmas so we can devide by it
    sigmas[sigmas == 0.0] = np.min(sigmas[sigmas != 0])

    # Lits start with a linear fit to logarithm of the data
    log_yx = np.log(ys)
    log_sigmas = np.array(sigmas) / np.log(ys)
    slope, intercept = fit_by_line(xs, log_yx, log_sigmas)
    central_value_guess = np.exp(intercept)
    exp_scale_guess = -1 / slope

    def objective(params):
        curve_prediced = explaw(params, xs)
        diff = (ys - curve_prediced) ** 2.0
        fitness = np.average(diff, weights=1/sigmas)
        return fitness
    bounds = ((central_value_guess/5, central_value_guess*5), (exp_scale_guess/5, exp_scale_guess*5))
    fit_result = differential_evolution(func=objective, bounds=bounds, polish=True)
    return fit_result.x


def gaussian(params, x):
    position = params[0]
    amplitude = params[1]
    stddev = params[2]
    dist = (x - position)
    dist[dist > math.pi] = 2*math.pi - dist[dist > math.pi]
    dist[dist < -math.pi] = -(2*math.pi - dist[dist < -math.pi])  # TODO: do this trick on FittingFunction class scale
    return amplitude * np.exp(-(dist / 4 / stddev)**2)


def gaussian_lopsided(params, x):
    position = params[0]
    amplitude = params[1]
    stddev = params[2]
    skew = params[3]
    dist = (x - position)
    res = np.zeros_like(x)
    neg_inds = np.where(dist < 0)
    pos_inds = np.where(dist >= 0)
    res[neg_inds] = amplitude * np.exp(-(dist[neg_inds] / 4 / stddev)**2)
    res[pos_inds] = amplitude * np.exp(-(dist[pos_inds] / 4 / (skew*stddev))**2)
    return res


def squeeze_angles(q, angles):
    """
    Function computes the values of angles after the coordinates plane is squeezed
    along the y-axis (i.e. after coordinate transformation x'=x_0, y'=q*y_0).
    """
    transformed_angles = []
    for alpha in angles:
        if math.pi/2 < alpha < 3*math.pi/2:
            x0 = -1
        else:
            x0 = 1
        alpha_new = math.atan2(q * x0 * math.tan(alpha), x0)
        if alpha_new < 0:
            alpha_new += 2 * math.pi
        transformed_angles.append(alpha_new)
    return transformed_angles
