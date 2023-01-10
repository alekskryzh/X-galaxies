#!/usr/bin/env python3

from os import path
import os
import sys
import math
import numpy as np
import yaml
import argparse
from matplotlib import pyplot as plt
from astropy.io import fits

from libs.ellipse import Ellipse
from libs.ellipse import fit_ellipse_to_data_de
from libs.ellipse import fit_ellipse_to_data_nm
from libs.ellipse import Point
from libs.fitters import GenericFitter
from libs.fitters import FittingFunction
from libs.output import ResultsFile
from libs.output import ResultShelve
from libs.output import LogFile
from libs.mathFunctions import fit_by_explaw
from libs.mathFunctions import explaw
from libs.mathFunctions import gaussian
from libs.mathFunctions import gaussian_lopsided
from libs.mathFunctions import squeeze_angles
plt.switch_backend('agg')


def main(args):
    results_dir = args.results_dir
    if not path.exists(results_dir):
        os.makedirs(results_dir)

    # Creating databases for results
    file_with_angles = ResultsFile(path.join(results_dir, "angles.dat"),
                                   column_names=["i1", "i2", "i3", "i4"])
    file_with_ellparams = ResultsFile(path.join(results_dir, "ellipses.dat"),
                                      column_names=["xcen", "ycen", "ell", "pa"], make_stats=False)
    file_with_intensities_of_peaks = ResultsFile(path.join(results_dir, "intensities.dat"),
                                                 column_names=["i1", "i2", "i3", "i4"])
    fitters_shelve = ResultShelve(path.join(results_dir, "fitters.db"))
    ellipses_shelve = ResultShelve(path.join(results_dir, "ellipses.db"))
    log_file = LogFile(path.join(results_dir, "logfile.dat"))

    # Load data
    data = fits.getdata(args.fits)

    log_file.write("Starting computation")
    # Obtain the reference ellipse
    if args.ref_ellipse is not None:
        # If the reference ellipse is provided in the form of shelve file,
        # we shall use it.
        log_file.write("Loading the reference ellipse from %s" % args.ref_ellipse)
        reference_ellipse = Ellipse.from_shelve(args.ref_ellipse)
    elif args.fix_ellipse is True:
        # Parameters of the ellipse are provided manually
        log_file.write("Creating ellipse with user-defined parameters", show=True)
        y_size, x_size = data.shape
        x_center = args.x_center if args.x_center > 0 else x_size / 2
        y_center = args.y_center if args.y_center > 0 else y_size / 2
        center_of_image = Point(x_center, y_center)
        reference_ellipse = Ellipse(radius=args.ref_radius, center=center_of_image,
                                    ellipticity=args.ell_ellipticity,
                                    posang=math.radians(args.ell_posang))
        # reference_ellipse.fit_center(data, verbose=args.verbose)
    else:
        # If the reference ellipse is *not* provided, then we should find it by fitting.
        # The reference ellipse will be fitted using most robust, but time consuming
        # techniques. The other ellipses will be fitted with faster algorithms
        # utilising the results of the reference ellipse as an initial condition
        log_file.write("Fitting an reference ellipse...")
        reference_ellipse = fit_ellipse_to_data_de(data, radius_of_ellipse=args.ref_radius,
                                                   list_of_fourier_modes=[4])
        reference_ellipse.save(path.join(results_dir, "ref_ellipse.db"))
    reference_ellipse.make_overlay_plot(data, path.join(results_dir, "ref_ellipse.png"))
    reference_ellipse.show()

    # Create some lists to be filled during the iteration over the radius
    all_peaks_cartesian_xs = []
    all_peaks_cartesian_ys = []

    # Create a list of functions to fit lobes (one function per lobe). Technically
    # it is possible to fit every lobe with a different function. But for now let us
    # set all the function to be the same (which is reasonable), only location bounds
    # to be different
    min_intensity = np.min(data)
    max_intensity = np.max(data)
    number_of_lobes = 4
    delta = 2 * math.pi / number_of_lobes
    lobe_centers = [delta/2 + i * delta for i in range(number_of_lobes)]
    # Take into account that our lobes are squeezed along y-axis a bit
    lobe_centers_ell = squeeze_angles(1-reference_ellipse.ellipticity, lobe_centers)
    lobe_width = math.pi / 4
    lobes_fitting_functions = []
    for lobe in range(number_of_lobes):
        if reference_ellipse.ellipticity == 1:
            if lobe < number_of_lobes / 2:
                loc_min = 0
                loc_max = math.pi
            else:
                loc_min = math.pi
                loc_max = 2 * math.pi
            widths = (0.001, 0.6)
        else:
            loc_min = lobe_centers_ell[lobe] - lobe_width - 0.25
            loc_max = lobe_centers_ell[lobe] + lobe_width + 0.25
            widths = (0.001, 0.6)
        bounds = [(loc_min, loc_max), (0, 1.25 * max_intensity), widths, (0.1, 5)]
        lobes_fitting_functions.append(FittingFunction(func=gaussian_lopsided, bounds=bounds,
                                                       tag="lobe_%i" % lobe))

    # Add disk
    bounds = [(-0.1, 0.1), (min_intensity, max_intensity), (0.001, 0.4)]
    lobes_fitting_functions.append(FittingFunction(func=gaussian, bounds=bounds, tag="disk_1"))
    bounds = [(math.pi-0.1, math.pi+0.1), (min_intensity, max_intensity), (0.001, 0.4)]
    lobes_fitting_functions.append(FittingFunction(func=gaussian, bounds=bounds, tag="disk_2"))

    # Leash disk intensities together
    leashed_parameters = [["disk_1", 1, "disk_2", 1]]  # intens param of 5th and 6th components are bound

    # Fit lobes for the reference ellipse:
    # Create a fitter object and feed the data to it
    exc_anomaly, observed_intensity, observed_intensity_std = reference_ellipse.get_inten_along_smooth(data)
    reference_fitter = GenericFitter(lobes_fitting_functions, leashed_parameters=leashed_parameters)
    reference_fitter.set_up_data(exc_anomaly, observed_intensity, observed_intensity_std)
    # Find intensities in several positions along ellipse to estimate bounds of parameters
    intens_0 = reference_ellipse.get_inten_at_angle(data, angle=0.0)
    intens_pi = reference_ellipse.get_inten_at_angle(data, angle=math.pi)
    intens_in_disk_plane = (intens_0 + intens_pi) / 2
    reference_fitter.update_parameter_bounds("disk_1", 1, 0.25*intens_in_disk_plane, intens_in_disk_plane)
    reference_fitter.update_parameter_bounds("disk_2", 1, 0.25*intens_in_disk_plane, intens_in_disk_plane)
    # Perform the fitting
    log_file.write("Performing the general fit on the reference ellipse.", show=True)
    reference_fitter.fit_azimuthal_slice_de()
    reference_fitter.show_fit()
    reference_fitter.extract_lobes()
    log_file.write("Fitting lobe for the reference ellipse.")
    ret_code = reference_fitter.fit_lobes_de()
    if reference_ellipse.ellipticity < 1:
        reference_fitter.make_plot(path.join(results_dir, "ref_ellipse_fit.png"))
    else:
        reference_fitter.make_plot(path.join(results_dir, "ref_ellipse_fit.png"), x_axis_range=data.shape[1])

    # Now we have the reference ellipse and its lobes fitted. We can proceed to
    # the fitting of the other ellipses. First we iterate from the reference
    # ellipse outwards untill we reach the args.max_radius value. After that
    # we go back to the reference ellipse and iterage inwards untill we reach
    # args.min_radius value
    initial_ellipse_params = reference_ellipse.get_parameters()
    # initial_general_fit_params = reference_fitter.fitted_params[:]
    initial_lobe_params = reference_fitter.lobe_fitted_parameters[:]
    log_file.write("Starting outward iteration", show=True)
    for radius in np.arange(args.ref_radius, args.max_radius+1, args.step):
        # Outwards iteration
        # Fit ellipse
        print("\n\n")
        log_file.write("Working with r=%i" % radius, show=True)
        if args.fix_ellipse is True:
            print("Parameters of the ellipse are provided manually and fixed")
            ellipse = Ellipse(radius=radius, center=center_of_image,
                              ellipticity=args.ell_ellipticity,
                              posang=math.radians(args.ell_posang))
        else:
            log_file.write("fitting the ellipse for", show=True)
            ellipse = fit_ellipse_to_data_nm(data, radius_of_ellipse=radius, list_of_fourier_modes=[4],
                                             initial_conditions=initial_ellipse_params, verbose=args.verbose)
        # Save ellipse as plot and as parameters
        ellipse.make_overlay_plot(data, path.join(results_dir, "ell_%i.png" % int(radius)))
        file_with_ellparams.write_data(radius, [ellipse.center.x, ellipse.center.y,
                                                ellipse.ellipticity, math.degrees(ellipse.posang)])

        exc_anomaly, observed_intensity, observed_intensity_std = ellipse.get_inten_along_smooth(data)

        # Creating fitter for the slise
        fitter = GenericFitter(lobes_fitting_functions, leashed_parameters=leashed_parameters)
        fitter.set_up_data(exc_anomaly, observed_intensity, observed_intensity_std)

        # Find intensities in several positions along ellipse to estimate bounds of parameters
        intens_0 = ellipse.get_inten_at_angle(data, angle=0.0)
        intens_pi = ellipse.get_inten_at_angle(data, angle=math.pi)
        intens_in_disk_plane = (intens_0 + intens_pi) / 2
        fitter.update_parameter_bounds("disk_1", 1, 0.25*intens_in_disk_plane, intens_in_disk_plane)
        fitter.update_parameter_bounds("disk_2", 1, 0.25*intens_in_disk_plane, intens_in_disk_plane)

        # Fitting the slice with the sum of four functions
        log_file.write("Performing general fit")
        ret_code = fitter.fit_azimuthal_slice_de(verbose=args.verbose)
        if ret_code != 0:
            print("Fitting failed")
            continue
        # fitter.fit_azimuthal_slice_nm(initial_general_fit_params, verbose=args.verbose)
        fitter.extract_lobes()

        # Fitting lobes
        log_file.write("Fitting lobes")
        ret_code = fitter.fit_lobes_de()
        if ret_code != 0:
            print("Fitting failed")
            continue
        # fitter.fit_lobes_gd(initial_lobe_params)
        if reference_ellipse.ellipticity < 1:
            fitter.make_plot(path.join(results_dir, "fit_%i.png" % int(radius)))
        else:
            fitter.make_plot(path.join(results_dir, "fit_%i.png" % int(radius)), x_axis_range=data.shape[1])

        # extract fitted positions of the lobes alon the slice
        lobe_positions = fitter.get_lobes_positions(verbose=args.verbose)
        # Find distances from peak point to the ellipse center
        distances = ellipse.get_distatnces(lobe_positions, data)
        mean_distance = np.mean(distances)
        # Find angles, the mean angle and the standart deviation
        lobe_angles = ellipse.get_angles(lobe_positions)
        file_with_angles.write_data(mean_distance, lobe_angles)
        # Find cartesian coordinates of peaks
        xs, ys = ellipse.get_xys(lobe_positions, data)
        all_peaks_cartesian_xs.extend(xs)
        all_peaks_cartesian_ys.extend(ys)
        # Save intensities of peaks
        i1, i2, i3, i4 = fitter.get_lobes_intensities()
        file_with_intensities_of_peaks.write_data(mean_distance, [i1, i2, i3, i4])

        # Save the ellipse and the fitter to databases
        ellipses_shelve.append(radius, ellipse)
        fitters_shelve.append(radius, fitter)

        # Extract fit results to use them as initial conditions for the next iterration
        initial_ellipse_params = ellipse.get_parameters()
        # initial_general_fit_params = fitter.fitted_params[:]
        initial_lobe_params = fitter.lobe_fitted_parameters[:]

    # Perform now the inwards iteration
    initial_ellipse_params = reference_ellipse.get_parameters()
    # initial_general_fit_params = reference_fitter.fitted_params[:]
    initial_lobe_params = reference_fitter.lobe_fitted_parameters[:]
    log_file.write("Starting inward iteration", show=True)
    for radius in np.arange(args.ref_radius-1, args.min_radius-1, -args.step):
        # Fit ellipse
        log_file.write("fitting the ellipse for r=%i" % radius, show=True)
        if args.fix_ellipse is True:
            print("Parameters of the ellipse are provided manually and fixed")
            ellipse = Ellipse(radius=radius, center=center_of_image,
                              ellipticity=args.ell_ellipticity,
                              posang=math.radians(args.ell_posang))
        else:
            log_file.write("fitting the ellipse for", show=True)
            ellipse = fit_ellipse_to_data_nm(data, radius_of_ellipse=radius, list_of_fourier_modes=[4],
                                             initial_conditions=initial_ellipse_params, verbose=args.verbose)
        # Save ellipse as plot and as parameters
        ellipse.make_overlay_plot(data, path.join(results_dir, "ell_%i.png" % int(radius)))
        file_with_ellparams.write_data(radius, [ellipse.center.x, ellipse.center.y,
                                                ellipse.ellipticity, math.degrees(ellipse.posang)])
        exc_anomaly, observed_intensity, observed_intensity_std = ellipse.get_inten_along_smooth(data)

        # Creating fitter for the slise
        fitter = GenericFitter(lobes_fitting_functions, leashed_parameters=leashed_parameters)
        fitter.set_up_data(exc_anomaly, observed_intensity, observed_intensity_std)

        # Find intensities in several positions along ellipse to estimate bounds of parameters
        intens_0 = ellipse.get_inten_at_angle(data, angle=0.0)
        intens_pi = ellipse.get_inten_at_angle(data, angle=math.pi)
        intens_in_disk_plane = (intens_0 + intens_pi) / 2
        fitter.update_parameter_bounds("disk_1", 1, 0.25*intens_in_disk_plane, intens_in_disk_plane)
        fitter.update_parameter_bounds("disk_2", 1, 0.25*intens_in_disk_plane, intens_in_disk_plane)

        # Fitting the slice with the sum of four functions
        log_file.write("Performing general fit")
        ret_code = fitter.fit_azimuthal_slice_de(verbose=args.verbose)
        if ret_code != 0:
            print("Failed fitting")
            continue
        fitter.extract_lobes()

        # Fitting lobes
        log_file.write("Fitting lobes")
        ret_code = fitter.fit_lobes_de(initial_lobe_params)
        if ret_code != 0:
            print("Fitting failed")
            continue
        if reference_ellipse.ellipticity < 1:
            fitter.make_plot(path.join(results_dir, "fit_%i.png" % int(radius)))
        else:
            fitter.make_plot(path.join(results_dir, "fit_%i.png" % int(radius)), x_axis_range=data.shape[1])

        # extract fitted positions of the lobes alon the slice
        lobe_positions = fitter.get_lobes_positions(verbose=args.verbose)
        # Find distances from peak point to the ellipse center
        distances = ellipse.get_distatnces(lobe_positions, data)
        mean_distance = np.mean(distances)
        # Find angles, the mean angle and the standart deviation
        lobe_angles = ellipse.get_angles(lobe_positions)
        file_with_angles.write_data(mean_distance, lobe_angles)
        # Find cartesian coordinates of peaks
        xs, ys = ellipse.get_xys(lobe_positions, data)
        all_peaks_cartesian_xs.extend(xs)
        all_peaks_cartesian_ys.extend(ys)

        # Save intensities of peaks
        i1, i2, i3, i4 = fitter.get_lobes_intensities()
        file_with_intensities_of_peaks.write_data(mean_distance, [i1, i2, i3, i4])

        # Save the ellipse and the fitter to databases
        ellipses_shelve.append(radius, ellipse)
        fitters_shelve.append(radius, fitter)

        # Extract fit results to use them as initial conditions for the next iterration
        initial_ellipse_params = ellipse.get_parameters()
        # initial_general_fit_params = fitter.fitted_params[:]
        initial_lobe_params = fitter.lobe_fitted_parameters[:]

    log_file.write("The iterations are done")

    # Perform analysis of the x-structure as a whole object (i.e. by
    # using the results of the fitting for all radii).

    # First of all let's plot peak positions over x-structure
    plt.figure(figsize=(5, 5), dpi=600)
    plt.imshow(data, origin='lower')
    plt.plot(all_peaks_cartesian_xs, all_peaks_cartesian_ys, 'ko', linestyle="", markersize=0.1)
    plt.savefig(path.join(results_dir, "peaks.png"))
    plt.clf()
    plt.close()

    # Plot angles of lobes as a function of the distance
    plt.figure(figsize=(5, 5), dpi=150)
    all_radii = file_with_angles.get_columns([0])
    colors = ["b", "g", "c", "y"]
    for lobe_number in range(0, 4):
        # Plot angles for all lobes
        angles = file_with_angles.get_columns([lobe_number+1])
        plt.plot(all_radii, angles, color=colors[lobe_number], linestyle=":")
    # Plot main angle and its std
    main_angles, main_angles_std = file_with_angles.get_columns([5, 6])
    plt.errorbar(x=all_radii, y=main_angles, yerr=main_angles_std, fmt="ro")
    plt.savefig(path.join(results_dir, "angles.png"))
    plt.clf()
    plt.close()

    # Fit peak intensities as a function of the distance by a line
    main_intensities, main_intensities_std = file_with_intensities_of_peaks.get_columns([5, 6])
    central_value, exp_scale = fit_by_explaw(all_radii, main_intensities, main_intensities_std)
    # Plot peak intensities as a function of the distance
    plt.figure(figsize=(5, 5), dpi=150)
    ax = plt.subplot(111)
    for lobe_number in range(0, 4):
        # Plot angles for all lobes
        intensities = file_with_intensities_of_peaks.get_columns([lobe_number+1])
        ax.plot(all_radii, intensities, color=colors[lobe_number], linestyle=":")
    # Plot main angle and its std
    ax.errorbar(x=all_radii, y=main_intensities, yerr=main_intensities_std, fmt="ro")
    # Plot line fit
    ax.plot(all_radii, explaw([central_value, exp_scale], all_radii), linestyle="solid")
    plt.annotate(s="h=%1.4f" % exp_scale, xy=(0.8, 0.8), xycoords="axes fraction")
    plt.xlabel("r")
    plt.ylabel("flux")
    plt.savefig(path.join(results_dir, "intensities.png"))
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    input_fits_name = sys.argv[1]
    config_file_name = sys.argv[2]
    with open(config_file_name) as config:
        params = yaml.load(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("fits", help="Input fits file")
    parser.add_argument("config", help="Config file name")
    for key, value in params.items():
        parser.add_argument("--%s" % key, default=value, type=type(value))
    args = parser.parse_args()
    main(args)
