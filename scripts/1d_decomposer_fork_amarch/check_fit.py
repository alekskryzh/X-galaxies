#!/usr/bin/env python

import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def main():
    image_data = fits.getdata(f"{sys.argv[1]}/image.fits")
    magzpt = float(fits.getheader(f"{sys.argv[1]}/image.fits")["MAGZPT"])
    mask_data = fits.getdata(f"{sys.argv[1]}/mask.fits")
    model_data = fits.getdata(f"{sys.argv[1]}/model.fits")
    image_data[mask_data != 0] *= np.nan

    y_size, x_size = image_data.shape
    x_cen = x_size // 2
    y_cen = y_size // 2

    width = 4
    data_slice = np.median(image_data[y_cen-width: y_cen+width, 0: -1], axis=0)
    x = np.arange(0, len(data_slice))
    data_slice[data_slice <= 1] = 1
    model_slice = np.median(model_data[y_cen-width: y_cen+width, 0: -1], axis=0)

    data_slice_mags = -2.5 * np.log10(data_slice / 0.25**2) + magzpt
    model_slice_mags = -2.5 * np.log10(model_slice / 0.25**2) + magzpt

    idx_bright = np.where((model_slice_mags < 26) * (data_slice_mags < 26))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x[idx_bright], data_slice_mags[idx_bright], "ro")
    ax.plot(x[idx_bright], model_slice_mags[idx_bright])
    plt.gca().invert_yaxis()
    plt.savefig(f"{sys.argv[1]}/slice.png")

if __name__ == '__main__':
    main()
