# xfit
A package for analysing parameters of X-structures using azimuthal slices along ellipses.

# Requirements
     - python 3.*
## Python packages:
     - yaml
	 - numpy
	 - scipy
	 - matplotlib
	 - astropy

# Usage
The script takes at least two command line arguments: a fits-file to analyse and a
name of a config file.

	./xfit.py fits_file.fits config.dat

Example of a config file (provided with the package):

    # A config file for xfit program
    # All parameters can be overwritten in the comand line with corresponging "--" key

    # A set of important parameters. Should be set carefully for every run
	ref_radius:  25       # Radius of the reference ellipse
	min_radius:  10       # Minimal radius to try to fit an ellipse
	max_radius:  75       # Maximal radius to try to fit an ellipse
	ref_ellipse: null     # Name of a file with the reference object or `null` if none given

    # Less important parameters. The defaults are probably fine
	verbose:     false    # Give more detailed information of the fit (false/true)
	results_dir: results  # Name of the directory for saving results

# Output
The output is stored in 'results_dir' directory and includes

	 - angles.dat -- an ASCII file with lobe angles as a function of distance to a galaxy centre
	 - angles.png -- a plot of a data in 'angles.dat' file
	 - ell*.png -- fitted ellipses showed over the galaxy image
	 - ellipses.dat -- parameters of fitted ellipses in ASCII format
	 - ellipses.db -- parameters of fitted ellipses serialised as Python shelve database
	 - fit*.png -- images of azimuthal slices fits
	 - fitters.db -- a database with fitting objects to reproduce fits
	 - intensities.dat -- intensities taken along lobes (ASCII)
	 - intensities.png -- plot of intensities.dat with the exponential law fit
	 - peaks.png -- peak positions plotted over the X-structure image
	 - ref_ellipse* -- files containing parameters of a reference ellipse
	 - logfile.dat
