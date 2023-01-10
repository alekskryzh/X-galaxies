#! /usr/bin/env python

from math import pi

import numpy as np
from scipy import odr
# from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt

plt.switch_backend('agg')


class FittingFunction(object):
    """
    Class stores a callable and some supplimentary data for using in
    the fitting process.
    """
    def __init__(self, func, bounds=None, tag="func"):
        """
        The first parameter of a fitting function has to govern the peak
        location, the second parameter -- peak intensity. All other parameters
        (if any can be in an arbitrary order).
        """
        self.func = func
        self.bounds = bounds
        self.npars = len(bounds)
        self.tag = tag

    def __call__(self, params, x):
        return self.func(params, x)


class GenericFitter(object):
    """
    A class to fit azimuthal (elliptic) slice with a given function
    """
    def __init__(self, fitting_functions, leashed_parameters=None):
        """
        Parameters:
           fitting_functions: a list of FittingFunction objects for
               fitting of every lobe.
               The first item of params list governs the angular
               location of the lobe on the elliptical slice. Rest of the
               parameters govern the shape of the lobe.
        """
        self.fitting_functions = fitting_functions
        self.fitting_functions_tags = [f.tag for f in fitting_functions]
        self.number_of_peaks = len(fitting_functions)  # Total number of peaks on the slice

        if leashed_parameters is None:
            self.leashed_parameters = []
        else:
            self.leashed_parameters = leashed_parameters

        # Since all parameters of all functions are fitted as a single vector,
        # they are stored in a common array. It is convenient to have a list
        # that contains indices of parameters for every function
        self.params_ranges = []
        param_start = 0
        for func in self.fitting_functions:
            param_end = param_start + func.npars
            self.params_ranges.append((param_start, param_end))
            param_start = param_end

        self.number_of_lobes = 4  # Number of x-structure lobes
        self.bounds = []
        for peak in range(self.number_of_peaks):
            self.bounds.extend(self.fitting_functions[peak].bounds)
        self.bounds.append((None, None))  # Bias bounds

        # Set some future objects to None for now
        self.fitted_params = None
        self.lobe_fitted_parameters = None
        self.lobe_anomalies = None
        self.lobe_initial_parameters = None

    def set_up_data(self, exc_anomaly, observed_intensity, observed_intensity_std):
        self.exc_anomaly = exc_anomaly
        self.observed_intensity = observed_intensity
        self.observed_intensity_std = observed_intensity_std
        self.number_of_data_points = len(exc_anomaly)
        self.anomaly_step = (exc_anomaly[-1] - exc_anomaly[0]) / self.number_of_data_points
        self.anomaly_before = np.linspace(-2*pi, 0, self.number_of_data_points)
        self.anomaly_after = np.linspace(2*pi, 4*pi, self.number_of_data_points)
        self.extended_anomaly = np.concatenate((self.anomaly_before, self.exc_anomaly, self.anomaly_after))
        self.bounds[-1] = (np.min(observed_intensity), np.median(observed_intensity))
        # Compute interpolation for the slice
        # extended_observed_intensity = np.concatenate(self.observed_intensity[self.number_of_data_points::-1],
        #                                              self.observed_intensity,
        #                                              self.observed_intensity[-1::-1])
        # self.interpolated_intensity = interp1d(x=self.extended_anomaly, y=extended_observed_intensity)

    def update_parameter_bounds(self, function_tag, param_idx, lower, upper):
        """
        Method sets new fitting ranges for a given parameter
        """
        func_idx = self.fitting_functions_tags.index(function_tag)
        param_start, _ = self.params_ranges[func_idx]
        self.bounds[param_start+param_idx] = (lower, upper)

    def compute_model_intensity(self, params, peaks_list=None, include_bias=True):
        extended_model_intensity = np.zeros_like(self.extended_anomaly)
        model_intensity = np.zeros_like(self.exc_anomaly)
        if peaks_list is None:
            peaks_list = range(self.number_of_peaks)
        for peak in peaks_list:
            peak_function = self.fitting_functions[peak]
            # Extract parameters for the current lobe from the list of all parameters
            param_start, param_end = self.params_ranges[peak]
            peak_params = params[param_start: param_end]
            # Calculate the intensity
            extended_model_intensity = peak_function(peak_params, self.extended_anomaly)
            # Copy the [0:2pi] range of the anomaly
            model_intensity += extended_model_intensity[self.number_of_data_points: -self.number_of_data_points]
            # Add the tail of extended anomaly to the beginning of the regular one
            tail = extended_model_intensity[-self.number_of_data_points:]
            model_intensity[: self.number_of_data_points] += tail[::-1]
            # Add the beginning of the extended anomaly to the end of the regular one
            head = extended_model_intensity[:self.number_of_data_points]
            model_intensity[-self.number_of_data_points:] += head[::-1]
        # Add bias
        if include_bias:
            model_intensity += params[-1]
        return model_intensity

    def fit_azimuthal_slice_de(self, verbose=False):
        """
        Function performs fit of the data using differential evolution algorithm
        """

        def fitness_function(params):
            model_intensity = self.compute_model_intensity(params)
            diff = (self.observed_intensity - model_intensity) ** 2
            fitness = np.average(diff, weights=1/self.observed_intensity_std)
            # Add leash penalty
            for i in range(len(self.leashed_parameters)):
                func_tag_1, par_num_1, func_tag_2, par_num_2 = self.leashed_parameters[i]
                func_num_1 = self.fitting_functions_tags.index(func_tag_1)
                func_num_2 = self.fitting_functions_tags.index(func_tag_2)
                param_start_1, _ = self.params_ranges[func_num_1]
                param_start_2, _ = self.params_ranges[func_num_2]
                value_1 = params[param_start_1+par_num_1]
                value_2 = params[param_start_2+par_num_2]
                penalty = abs(value_1 - value_2)
                fitness += penalty
            return fitness

        self.fit_result = differential_evolution(fitness_function, self.bounds, popsize=30, polish=False)
        self.fitted_params = self.fit_result.x
        self.store_models()
        if verbose is True:
            self.show_fit()
        if not all(np.isfinite(self.fitted_params)):
            print("Nan values occured during general DE fit")
            return 1
        # if not self.fit_result.success:
        #     print("DE returned 'False' as success value")
        #     return 1
        # Check if the fit converged to the bounds limit for lobes
        num_of_lobes_parameters = self.number_of_lobes * len(self.fitting_functions[0].bounds)
        for idx in range(num_of_lobes_parameters):
            bound_width = self.bounds[idx][1] - self.bounds[idx][0]
            if (self.bounds[idx][0] != 0) and (self.fitted_params[idx] - self.bounds[idx][0] < 0.01 * bound_width):
                print("General DE problem: parameter %i is too close to the lower bound" % idx)
                return 1
            if (self.bounds[idx][1] - self.fitted_params[idx]) < 0.01 * bound_width:
                print("General DE problem: parameter %i is too close to the upper bound" % idx)
                return 1
        return 0

    def fit_azimuthal_slice_nm(self, initial_conditions, verbose=False):
        """
        The same as fit_azimuthal_slice_de, but using Nelder-Mead simples method
        instead of differential evolution. Requires initial conditions for the fit.
        """
        def fitness_function(params):
            model_intensity = self.compute_model_intensity(params)
            diff = (self.observed_intensity - model_intensity) ** 2
            fitness = np.average(diff, weights=1/self.observed_intensity_std)
            return fitness

        self.fit_result = minimize(fun=fitness_function, x0=initial_conditions,
                                   options={"disp": verbose}, method="Nelder-Mead",
                                   bounds=self.bounds).x

        # Show fit results if verbose is True
        self.fitted_params = self.fit_result
        self.store_models()
        if verbose is True:
            self.show_fit()

    def show_fit(self):
        """
        Method prints out fitting parameters of all peaks
        """
        print("Fit result:")
        for peak in range(self.number_of_peaks):
            print("Lobe %i : " % peak, end='')
            param_start, param_end = self.params_ranges[peak]
            print(self.fitted_params[param_start: param_end])
        print("Bias: %1.2f" % self.fitted_params[-1])

    def store_models(self):
        """
        Function computes models of slice (full, disk, x-structure)
        and stores it inside self
        """
        peaks_list = range(self.number_of_lobes, self.number_of_peaks)
        self.model_intensity_total = self.compute_model_intensity(self.fitted_params)
        self.model_intensity_disk = self.compute_model_intensity(self.fitted_params,
                                                                 peaks_list=peaks_list,
                                                                 include_bias=False)
        self.model_intensity_xstructure = self.compute_model_intensity(self.fitted_params,
                                                                       peaks_list=range(self.number_of_lobes))

    def make_plot(self, file_name=None, x_axis_range=None):
        """
        When file_name is None shows the plot.
        The plot consists of two pannels: the upper one shows the original slice
        of the data, and the lower one shows the disk-subtracted slice.
        """
        plt.figure(figsize=(5, 10))
        # Upper panel
        plt.subplot(211)
        if x_axis_range is None:
            coeff = 1
        else:
            coeff = x_axis_range / pi
        # Plot data
        plt.fill_between(coeff * self.exc_anomaly, y1=self.observed_intensity-self.observed_intensity_std,
                         y2=self.observed_intensity+self.observed_intensity_std, color="k", alpha=0.3)
        plt.plot(coeff * self.exc_anomaly, self.observed_intensity, color="b")
        # Plot full global fit
        plt.plot(coeff * self.exc_anomaly, self.model_intensity_total, linestyle=":", color="m")
        # Plot disk model
        plt.plot(coeff * self.exc_anomaly, self.model_intensity_disk, linestyle=":", color="g")
        # plot lobes model
        plt.plot(coeff * self.exc_anomaly, self.model_intensity_xstructure, linestyle=":", color="g")

        # Lower panel
        plt.subplot(212)
        # Plot disk-subtracted data
        y1 = self.observed_intensity-self.observed_intensity_std-self.model_intensity_disk
        y2 = self.observed_intensity+self.observed_intensity_std-self.model_intensity_disk
        plt.fill_between(x=coeff*self.exc_anomaly, y1=y1, y2=y2, color="k", alpha=0.3)
        plt.plot(coeff * self.exc_anomaly, self.observed_intensity-self.model_intensity_disk, color="b")

        # Plot disk-subtracred fit
        plt.plot(coeff * self.exc_anomaly, self.model_intensity_xstructure, linestyle=":", color="m")

        # # Plot selected lobes
        bias = self.fitted_params[-1]
        if self.lobe_anomalies is not None:
            for i in range(self.number_of_lobes):
                plt.plot(coeff * self.lobe_anomalies[i], self.lobe_intensities[i]+bias, "ro", markersize=1)

        # Plot individual fits of lobes
        if self.lobe_fitted_parameters:
            for i in range(self.number_of_lobes):
                lobe_fitted_intensity = self.fitting_functions[i](self.lobe_fitted_parameters[i],
                                                                  self.lobe_anomalies[i]) + bias
                plt.plot(coeff * self.lobe_anomalies[i], lobe_fitted_intensity, 'go', markersize=1)

        # Show or save a plot
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
        plt.cla()
        plt.close()

    def extract_lobes(self, width=None, height=0.5):
        """
        Method extracts parts of the azimuthal slice that correspond to the
        separate lobes
        """
        self.lobe_anomalies = []
        self.lobe_intensities = []
        self.lobe_intensities_std = []
        # Take initial guesses for lobe individual fits from the global fit
        self.lobe_initial_parameters = []
        # The lobe is the peak between two minimums
        bias = self.fitted_params[-1]
        for peak in range(self.number_of_lobes):
            param_start, param_end = self.params_ranges[peak]
            peak_position = self.fitted_params[param_start]
            dist = np.abs(self.exc_anomaly-peak_position)
            inds_left_of_peak = np.where((self.exc_anomaly < peak_position) * (dist < pi/4))[0]
            inds_rigth_of_peak = np.where((self.exc_anomaly > peak_position) * (dist < pi/4))[0]

            idx_left_min = np.argmin(self.observed_intensity[inds_left_of_peak]) + inds_left_of_peak[0]
            idx_right_min = np.argmin(self.observed_intensity[inds_rigth_of_peak]) + inds_rigth_of_peak[0]
            lobe_inds = list(range(idx_left_min, idx_right_min))

            self.lobe_anomalies.append(self.exc_anomaly[lobe_inds])
            # Extract the lobe from the entire slice and subtract the bias term
            lobe_intens = self.observed_intensity[lobe_inds] - bias - self.model_intensity_disk[lobe_inds]
            self.lobe_intensities.append(lobe_intens)
            self.lobe_intensities_std.append(self.observed_intensity_std[lobe_inds])
            self.lobe_initial_parameters.append(self.fitted_params[param_start:param_end])

    # def extract_lobes(self, width=None, height=0.5):
    #     """
    #     Method extracts parts of the azimuthal slice that correspond to the
    #     separate lobes
    #     """
    #     self.lobe_anomalies = []
    #     self.lobe_intensities = []
    #     self.lobe_intensities_std = []
    #     # Take initial guesses for lobe individual fits from the global fit
    #     self.lobe_initial_parameters = []
    #     # Let's define the lobe as a part of the azimuthal slice that
    #     # lies close to the fitted peak center and has the intensity
    #     # higher than a specified threshold
    #     bias = self.fitted_params[-1]
    #     for peak in range(self.number_of_lobes):
    #         param_start, param_end = self.params_ranges[peak]
    #         peak_position = self.fitted_params[param_start]
    #         if width is None:
    #             width = 2*self.fitted_params[param_start+2]
    #         # Find the part of the slice that lies close to the peak
    #         dist = np.abs(self.exc_anomaly-peak_position)
    #         inds_close_to_peak = np.where((dist < width/2) | (dist > (2*pi-width/2)))

    #         # Find index of maximum
    #         idx_max = np.argmin(dist)
    #         # now let's go to the left and to the right untill we reach height*max_flux value
    #         amplitude = np.max(self.model_intensity_xstructure[inds_close_to_peak])
    #         thresh = max(height * amplitude, bias)

    #         for j in range(idx_max, 0, -1):
    #             if (self.observed_intensity[j] < thresh) or (dist[j] > width * 3):
    #                 idx_dist_to_left_bound = (idx_max-j)
    #                 break
    #         for j in range(idx_max, len(self.observed_intensity)):
    #             if (self.observed_intensity[j] < thresh) or (dist[j] > width * 3):
    #                 idx_dist_to_right_bound = (j-idx_max)
    #                 break
    #         idx_mean_width = int((idx_dist_to_left_bound+idx_dist_to_right_bound) / 2)
    #         if idx_mean_width < 5:
    #             idx_mean_width = 5
    #         lobe_inds = list(range(idx_max-idx_mean_width, idx_max+idx_mean_width))

    #         self.lobe_anomalies.append(self.exc_anomaly[lobe_inds])
    #         # Extract the lobe from the entire slice and subtract the bias term
    #         lobe_intens = self.observed_intensity[lobe_inds] - bias - self.model_intensity_disk[lobe_inds]
    #         self.lobe_intensities.append(lobe_intens)
    #         self.lobe_intensities_std.append(self.observed_intensity_std[lobe_inds])
    #         self.lobe_initial_parameters.append(self.fitted_params[param_start:param_end])

    def fit_lobes_gd(self, initial_conditions=None, verbose=False):
        """
        Method perfomrs fit of the fitting function separately in each
        lobe using the ODR method and the results of the general fit
        as a starting point
        """
        if verbose:
            print("Fitting lobes with gd algorithm")
        self.lobe_fitted_parameters = []  # Results of lobe individual fits
        for peak in range(self.number_of_lobes):
            model = odr.Model(self.fitting_functions[peak])
            data = odr.Data(x=self.lobe_anomalies[peak],
                            y=self.lobe_intensities[peak],
                            we=1/self.lobe_intensities_std[peak])
            if initial_conditions is None:
                beta0 = self.lobe_initial_parameters[peak]
            else:
                beta0 = initial_conditions[peak]
            fitter = odr.ODR(data=data, model=model, beta0=beta0)
            lobe_fit_result = fitter.run()
            if verbose:
                print("Peak %i " % peak, lobe_fit_result.beta)
            self.lobe_fitted_parameters.append(lobe_fit_result.beta)

    def fit_lobes_de(self, verbose=False):
        """
        Method pefroms fit of the fitting function separately in each
        lobe using the DE method.
        """
        ret_code = 0
        if verbose:
            print("Fitting lobes with DE algorithm")
        self.lobe_fitted_parameters = []  # Results of lobe individual fits
        for lobe in range(self.number_of_lobes):
            def fitness_function(params):
                model_intensity = self.fitting_functions[lobe](params, self.lobe_anomalies[lobe])
                diff = (self.lobe_intensities[lobe] - model_intensity) ** 2
                fitness = np.average(diff, weights=1/self.lobe_intensities_std[lobe])
                return fitness
            bounds = self.fitting_functions[lobe].bounds
            lobe_fit_result = differential_evolution(fitness_function,
                                                     bounds=bounds)
            # Check if the fit converged to the bounds limit
            for idx in range(len(bounds)):
                bound_width = bounds[idx][1] - self.bounds[idx][0]
                if (bounds[idx][0] != 0) and (lobe_fit_result.x[idx] - bounds[idx][0] < 0.01 * bound_width):
                    print("Lobe DE problem: parameter %i of lobe %i is too close to the lower bound" % (idx, lobe))
                    print("Its value is %1.4f while bounds are %1.4f -- %1.4f" % (lobe_fit_result.x[idx],
                                                                                  bounds[idx][0], bounds[idx][1]))
                    ret_code = 1
                if (bounds[idx][1] - lobe_fit_result.x[idx]) < 0.01 * bound_width:
                    print("Lobe DE problem: parameter %i of lobe %i is too close to the upper bound" % (idx, lobe))
                    print("Its value is %1.4f while bounds are %1.4f -- %1.4f" % (lobe_fit_result.x[idx],
                                                                                  bounds[idx][0], bounds[idx][1]))
                    ret_code = 1
            self.lobe_fitted_parameters.append(lobe_fit_result.x)
            if verbose:
                print("Lobe fit:", lobe_fit_result.x)
        return ret_code

    def get_lobes_positions(self, verbose=False):
        """
        Return values of position parameters of lobes. If lobes fitting was performed,
        then the fitted results are used; otherwise the results of the general fit is
        used as the estimate of the lobes location
        """
        positions = []
        if self.lobe_fitted_parameters is not None:
            for lobe in range(self.number_of_lobes):
                positions.append(self.lobe_fitted_parameters[lobe][0])
        elif self.lobe_initial_parameters is not None:
            for lobe in range(self.number_of_lobes):
                positions.append(self.lobe_initial_parameters[lobe][0])
        else:
            print("Lobes weren't fitted nor extracted")
            return None
        if verbose:
            print("positions", positions)
        return positions

    def get_lobes_intensities(self):
        """
        Method returns peak intensities of lobes based on their fit by a function
        """
        bias = self.fitted_params[-1]
        return [par[1]+bias for par in self.lobe_fitted_parameters[:self.number_of_lobes]]
