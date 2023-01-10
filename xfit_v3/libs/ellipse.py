#! /usr/bin/env python

from math import pi
from math import sin
from math import cos
import math
import os
from os import path
import shelve
import numpy as np
from scipy.optimize import minimize
from matplotlib.patches import Ellipse as MatPlotEllipse
from scipy.optimize import differential_evolution
from photutils import isophote
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Ellipse(object):
    def __init__(self, radius, center, ellipticity, posang):
        self.radius = radius
        self.center = center
        self.ellipticity = ellipticity
        self.q_value = 1-self.ellipticity
        self.posang = posang
        self.fourier_modes = []
        self.fourier_amplitudes = []
        self.fourier_phases = []

    @classmethod
    def from_fit_results(cls, fit_result, radius, fourier_modes):
        x_cen = fit_result[0]
        y_cen = fit_result[1]
        ellipticity = fit_result[2]
        posang = fit_result[3]
        center = Point(x_cen, y_cen)
        ell = cls(radius=radius, center=center, ellipticity=ellipticity, posang=posang)
        for i, n in enumerate(fourier_modes):
            amplitude = fit_result[4+2*i]
            phase = fit_result[4+2*i+1]
            ell.add_fourier_mode(n, amplitude, phase)
        return ell

    @classmethod
    def from_shelve(cls, file_name):
        """
        Load the ellipse object from a shelve file
        """
        if not path.exists(file_name):
            print("Error [from_shelve method]: file %s not found" % file_name)
            exit(1)
        db = shelve.open(file_name)
        obj = db["ellipse"]
        db.close()
        return obj

    def save(self, file_name):
        """
        Save the current object to a shelve file
        """
        if path.exists(file_name):
            os.remove(file_name)
        db = shelve.open(file_name)
        db["ellipse"] = self
        db.close()

    def show(self):
        print("Ellipse:")
        print("    Center: %1.2f  %1.2f  [pix]" % (self.center.x, self.center.y))
        print("    radius: %1.2f [pix]" % self.radius)
        print("    ell: %1.2f" % self.ellipticity)
        print("    posang: %1.2f deg" % math.degrees(self.posang))
        for i, n in enumerate(self.fourier_modes):
            print("    Fourier[%i]: %1.2f  %1.2f" % (n, self.fourier_amplitudes[i], self.fourier_phases[i]))

    def get_parameters(self):
        """
        Function returns all parameters of the ellipse including the Fourier
        modes to use them as initial conditions for the fitting of the
        adjacent ellipse via the gradient descent algorithm
        """
        params = []
        params.append(self.center.x)
        params.append(self.center.y)
        params.append(self.ellipticity)
        params.append(self.posang)
        for i in range(len(self.fourier_modes)):
            params.append(self.fourier_amplitudes[i])
            params.append(self.fourier_phases[i])
        return params

    def add_fourier_mode(self, number, amplitude, phase):
        self.fourier_modes.append(number)
        self.fourier_amplitudes.append(amplitude)
        self.fourier_phases.append(phase)

    def get_inten_along(self, data):
        exc_anomaly = np.linspace(0, 2*pi, 10*self.radius)
        intensity = np.zeros_like(exc_anomaly)
        sinpa = sin(self.posang)
        cospa = cos(self.posang)
        cose = np.cos(exc_anomaly)
        sine = np.sin(exc_anomaly)
        for i in range(len(exc_anomaly)):
            fx, ix = math.modf(self.center.x + self.radius * cose[i] * cospa -
                               self.radius * self.q_value * sine[i] * sinpa)
            fy, iy = math.modf(self.center.y + self.radius * self.q_value * sine[i] * cospa +
                               self.radius * cose[i] * sinpa)
            iix = int(ix)
            iiy = int(iy)
            intensity[i] = ((1.0-fx)*(1.0-fy)*data[iiy][iix] + fx*(1.0-fy)*data[iiy][iix+1] +
                            fy*(1.0-fx)*data[iiy+1][iix] + fx*fy*data[iiy+1][iix+1])
        return exc_anomaly, intensity

    def get_inten_along_smooth(self, data):
        """
        Method makes an azimuthal slice along the ellipse. The smoothing is
        performed by the averaging of several ellipses with a bit different radii
        """
        if self.ellipticity == 1.0:
            return self.get_inten_along_square(data)
        exc_anomaly = np.linspace(0, 2*pi, 10*self.radius)
        intensity = np.zeros_like(exc_anomaly)
        intensity_std = np.zeros_like(exc_anomaly)
        sinpa = sin(self.posang)
        cospa = cos(self.posang)
        cose = np.cos(exc_anomaly)
        sine = np.sin(exc_anomaly)
        y_size, x_size = data.shape
        for i in range(len(exc_anomaly)):
            i_along_r = []
            for r in np.linspace(self.radius-2, self.radius+2, 5):
                fx, ix = math.modf(self.center.x + r * cose[i] * cospa - r * self.q_value * sine[i] * sinpa)
                fy, iy = math.modf(self.center.y + r * self.q_value * sine[i] * cospa + r * cose[i] * sinpa)
                iix = int(ix)
                iiy = int(iy)
                if (iix < 1) or (iiy < 1) or (iix >= x_size-1) or (iiy >= y_size-1):
                    # We are outside of the image
                    i_along_r.append(0.0)
                else:
                    i_along_r.append(((1.0-fx)*(1.0-fy)*data[iiy][iix] + fx*(1.0-fy)*data[iiy][iix+1] +
                                      fy*(1.0-fx)*data[iiy+1][iix] + fx*fy*data[iiy+1][iix+1]))
            intensity[i] = np.median(i_along_r)
            intensity_std[i] = np.std(i_along_r)
        return exc_anomaly, intensity, intensity_std

    def get_inten_along_square(self, data):
        """
        This function is for making a slices parallel to the galaxy disk.
        excenomaly is not a real elliptical exc. anomaly but rather a linear position
        on the slice measured in relative values such that values 0..pi are for the
        half-slice that is above the galaxy and pnes pi..2pi are for a half-slice
        that is below the galaxy disk.
        """
        y_size, x_size = data.shape
        exc_anomaly = []
        intensity = []
        intensity_std = []
        # Compute the half-slice that is above the galaxy
        for x_slice in np.arange(0, x_size):
            intens_for_smoothing = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    fx, ix = math.modf(x_slice + i)
                    fy, iy = math.modf(self.center.y + self.radius + j)
                    iix = int(ix)
                    iiy = int(iy)
                    if (iix < 1) or (iiy < 1) or (iix >= x_size-1) or (iiy >= y_size-1):
                        # We are outside of the image
                        intens_for_smoothing.append(0.0)
                    else:
                        intens_for_smoothing.append(((1.0-fx)*(1.0-fy)*data[iiy][iix] + fx*(1.0-fy)*data[iiy][iix+1] +
                                                     fy*(1.0-fx)*data[iiy+1][iix] + fx*fy*data[iiy+1][iix+1]))
            intensity.append(np.median(intens_for_smoothing))
            intensity_std.append(np.std(intens_for_smoothing))
            exc_anomaly.append(math.pi * x_slice/x_size)

        # Compute below galaxy half-slice
        for x_slice in np.arange(0, x_size):
            intens_for_smoothing = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    fx, ix = math.modf(x_slice + i)
                    fy, iy = math.modf(self.center.y - self.radius + j)
                    iix = int(ix)
                    iiy = int(iy)
                    if (iix < 1) or (iiy < 1) or (iix >= x_size-1) or (iiy >= y_size-1):
                        # We are outside of the image
                        intens_for_smoothing.append(0.0)
                    else:
                        intens_for_smoothing.append(((1.0-fx)*(1.0-fy)*data[iiy][iix] + fx*(1.0-fy)*data[iiy][iix+1] +
                                                     fy*(1.0-fx)*data[iiy+1][iix] + fx*fy*data[iiy+1][iix+1]))
            intensity.append(np.median(intens_for_smoothing))
            intensity_std.append(np.std(intens_for_smoothing))
            exc_anomaly.append(math.pi + math.pi * x_slice/x_size)
        return np.array(exc_anomaly), np.array(intensity), np.array(intensity_std)

    def get_inten_at_angle(self, data, angle):
        """
        Method returns the intensity at an anomaly value equal to
        the given angle
        """
        if self.ellipticity == 1.0:
            return self.get_inten_at_angle_square(data, angle)
        y_size, x_size = data.shape
        fx, ix = math.modf(self.center.x + self.radius * cos(angle) * cos(self.posang)
                           - self.radius * self.q_value * sin(angle) * sin(self.posang))
        fy, iy = math.modf(self.center.y + self.radius * self.q_value * sin(angle) * cos(self.posang)
                           + self.radius * cos(angle) * sin(self.posang))
        iix = int(ix)
        iiy = int(iy)
        if (iix < 1) or (iiy < 1) or (iix >= x_size-1) or (iiy >= y_size-1):
            # We are outside of the image
            inten = 0.0
        else:
            inten = ((1.0-fx)*(1.0-fy)*data[iiy][iix] + fx*(1.0-fy)*data[iiy][iix+1] +
                     fy*(1.0-fx)*data[iiy+1][iix] + fx*fy*data[iiy+1][iix+1])
        return inten

    def get_inten_at_angle_square(self, data, angle):
        """
        Method returns some values for 'flat ellipse' approximation
        """
        y_size, x_size = data.shape
        if angle < math.pi:
            x = x_size * angle / math.pi
            y = self.center.y + self.radius
        else:
            x = x_size * (angle-math.pi) / math.pi
            y = self.center.y - self.radius
        fx, ix = math.modf(x)
        fy, iy = math.modf(y)
        iix = int(ix)
        iiy = int(iy)
        if (iix < 1) or (iiy < 1) or (iix >= x_size-1) or (iiy >= y_size-1):
            # We are outside of the image
            inten = 0.0
        else:
            inten = ((1.0-fx)*(1.0-fy)*data[iiy][iix] + fx*(1.0-fy)*data[iiy][iix+1] +
                     fy*(1.0-fx)*data[iiy+1][iix] + fx*fy*data[iiy+1][iix+1])
        return inten

    def make_overlay_plot(self, data, file_name=None):
        """
        Method adds a plot with an overlay of the ellipse over the data array to the given axis object.
        When file_name is None shows the plot instead of saving it.
        """
        plt.figure(figsize=(5, 5), dpi=500)
        ax = plt.subplot(111)
        x_size, y_size = data.shape
        plt.imshow(data, aspect='equal', origin='lower', extent=(0, y_size, 0, x_size))
        if self.ellipticity < 1.0:
            xy = (self.center.x, self.center.y)
            width = 2*self.radius
            height = 2*self.radius*(1-self.ellipticity)
            angle = math.degrees(self.posang)
            el = MatPlotEllipse(xy=xy, width=width, height=height, angle=angle, fill=False, ec="k",
                                linewidth=0.35)
            ax.add_patch(el)
        else:
            plt.axhline(y=self.center.y+self.radius)
            plt.axhline(y=self.center.y-self.radius)

        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
        plt.cla()
        plt.close()

    def make_slice_plot(self, data, file_name=None):
        """
        Method makes a slice of data along the ellipse and plot it
        """
        exc_anomaly, intensity, intensity_std = self.get_inten_along_smooth(data)
        plt.figure(figsize=(5, 5), dpi=500)
        plt.subplot(111)
        plt.plot(exc_anomaly, intensity, color="k")
        plt.fill_between(exc_anomaly, y1=intensity-intensity_std, y2=intensity+intensity_std,
                         color="k", alpha=0.3)
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
        plt.cla()
        plt.close()

    def get_fitness(self, data):
        """
        Function shows how good this ellipse fits the data isophote
        """
        exc_anomaly, observed_intensity, observed_intensity_std = self.get_inten_along_smooth(data)
        # Add fourier component now
        median_intensity = np.median(observed_intensity)
        fourier_component = np.zeros_like(observed_intensity)
        for i, n in enumerate(self.fourier_modes):
            fourier_component += self.fourier_amplitudes[i] * median_intensity *\
                                 np.cos(n * exc_anomaly + self.fourier_phases[i])
        model_intensity = median_intensity + fourier_component
        diff = (observed_intensity - model_intensity) ** 2
        fitness = np.average(diff, weights=1/observed_intensity_std)
        return fitness

    def get_xys(self, anomaly, data=None):
        """
        Method computes the cartesian coordinates of the point with the given anomaly
        on the ellipse.
        """
        if self.ellipticity == 1:
            return self.get_xys_square(anomaly, data)
        x = self.center.x + self.radius * np.cos(anomaly) * cos(self.posang) - \
            self.radius * self.q_value * np.sin(anomaly) * sin(self.posang)
        y = self.center.y + self.radius * self.q_value * np.sin(anomaly) * cos(self.posang) + \
            self.radius * np.cos(anomaly) * sin(self.posang)
        return x, y

    def get_xys_square(self, angle, data):
        y_size, x_size = data.shape
        angle = np.array(angle)
        x = np.zeros_like(angle)
        y = np.zeros_like(angle)
        x[angle < math.pi] = x_size * angle[angle < math.pi] / math.pi
        y[angle < math.pi] = self.center.y + self.radius
        x[angle > math.pi] = x_size * (angle[angle > math.pi]-math.pi) / math.pi
        y[angle > math.pi] = self.center.y - self.radius
        return x, y

    def get_angles(self, anomaly):
        """
        Returns the azimuthal angle in the polar coordinate system of the point
        on the ellipse with the given anomaly
        """
        if self.ellipticity == 1:
            return anomaly
        # Since we are interested only in the opening angle of lobes, we should
        # ignore the position angle of the ellipse, since it only governs the
        # rotation of the whole structure, but not the opening angle.
        # So posang = 0  =>  cos(posang) = 1 and sin(posang) = 0.
        # Also the angle does not depend on the radius of the ellipse
        x = np.cos(anomaly)
        y = self.q_value * np.sin(anomaly)
        angle = np.degrees(np.arctan2(np.abs(y), np.abs(x)))
        return angle

    def get_distatnces(self, anomaly, data=None):
        """
        Return distanges from points on the ellipse with given anomalies
        to the ellipse centre
        """
        if self.ellipticity == 1.0:
            x, y = self.get_xys_square(anomaly, data)
            x = x - self.center.x
            y = y - self.center.y
        x = self.radius * np.cos(anomaly) * cos(self.posang) - \
            self.radius * self.q_value * np.sin(anomaly) * sin(self.posang)
        y = self.radius * self.q_value * np.sin(anomaly) * cos(self.posang) + \
            self.radius * np.cos(anomaly) * sin(self.posang)
        return np.hypot(x, y)

    def fit_center(self, data, verbose=False):
        """
        Method finds best-fit center position of the ellipse using photutils
        """
        geometry = isophote.EllipseGeometry(x0=self.center.x, y0=self.center.y, sma=self.radius,
                                            eps=self.ellipticity, pa=self.posang)
        geometry.find_center(data, verbose=verbose)
        self.center.x = geometry.x0
        self.center.y = geometry.y0


def fit_ellipse_to_data_de(data, radius_of_ellipse, list_of_fourier_modes):
    """
    Fit an ellipse of a given radius with given fourier modes into data
    using the differential evolution method
    """
    # Create function to minimize.
    def objective(params):
        """
        This function is to be minimized. It creates an ellipse with given
        parameters and returns fitness score of the ellipse.
        """
        x_cen = params[0]
        y_cen = params[1]
        ellipticity = params[2]
        posang = params[3]
        center = Point(x_cen, y_cen)
        ell = Ellipse(radius=radius_of_ellipse, center=center, ellipticity=ellipticity, posang=posang)
        for i, n in enumerate(list_of_fourier_modes):
            amplitude = params[4+2*i]
            phase = params[4+2*i+1]
            ell.add_fourier_mode(n, amplitude, phase)
        return ell.get_fitness(data)

    # Create bounds for the differential evolution minimizer
    y_size, x_size = data.shape  # Suppose the galaxy is centered on the image center
    x_center_approx = x_size / 2
    y_center_approx = y_size / 2
    bounds = [(x_center_approx-3, x_center_approx+3)]      # x-coordinate
    bounds.append((y_center_approx-3, y_center_approx+3))  # y-coordinate
    bounds.append((0.0, 0.85))  # ellipticity
    bounds.append((-0.3, 0.3))  # position angle
    for i in list_of_fourier_modes:
        # bounds for fourier modes
        bounds.append((-2, 2))
        bounds.append((-pi/2, pi/2))
    fit_result = differential_evolution(objective, bounds=bounds)
    fitted_ellipse = Ellipse.from_fit_results(fit_result.x, radius_of_ellipse, list_of_fourier_modes)
    return fitted_ellipse


def fit_ellipse_to_data_nm(data, radius_of_ellipse, list_of_fourier_modes, initial_conditions, verbose=False):
    """
    Fit an ellipse of a given radius with given fourier modes into data
    using the Nelder-Mead simplex method and the initial conditions from the
    fit of another ellipse
    """
    # Create function to minimize.
    def objective(params):
        """
        This function is to be minimized. It creates an ellipse with given
        parameters and returns fitness score of the ellipse.
        """
        x_cen = params[0]
        y_cen = params[1]
        ellipticity = params[2]
        posang = params[3]
        center = Point(x_cen, y_cen)
        ell = Ellipse(radius=radius_of_ellipse, center=center, ellipticity=ellipticity, posang=posang)
        for i, n in enumerate(list_of_fourier_modes):
            amplitude = params[4+2*i]
            phase = params[4+2*i+1]
            ell.add_fourier_mode(n, amplitude, phase)
        return ell.get_fitness(data)

    fit_result = minimize(fun=objective, x0=initial_conditions, options={"disp": verbose},
                          method="Nelder-Mead").x
    fitted_ellipse = Ellipse.from_fit_results(fit_result, radius_of_ellipse, list_of_fourier_modes)  # TODO: bounds
    return fitted_ellipse
