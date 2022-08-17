#!/usr/bin/env python

import argparse
from pathlib import Path
import os
import math
import shutil
import time
import json
import subprocess
import warnings
import socket
from string import Template

from scipy.ndimage import zoom
from scipy.optimize import fmin
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.modeling import models, fitting

from libs.fitters import SimpleDiskFitter
from libs.fitters import BrokenDiskFitter
from libs.fitters import DoubleBrokenDiskFitter
from libs.fitters import BulgeSimpleDiskFitter
from libs.fitters import BulgeBrokenDiskFitter
from libs.fitters import BulgeDoubleBrokenDiskFitter
from libs.fitters import VerticalDiskFitter
from libs.fitters import AddVerticalDiskFitter
from libs.fitters import BulgeVerticalDiskFitter

warnings.filterwarnings("ignore")

matplotlib.rcParams['keymap.fullscreen'].remove("f")
matplotlib.rcParams['keymap.save'].remove("s")
matplotlib.rcParams['keymap.back'].remove('left')
matplotlib.rcParams['keymap.forward'].remove('right')

matplotlib.rcParams['text.usetex'] = False

workdir = Path("workdir")
kappa = 2.5 / math.log(10)
image_scale = 0.262  # arcsecs per pixel
colors = ["b", "g", "k"]


def fit_2dgaussian(data):
    y_size, x_size = data.shape
    p_init = models.Gaussian2D(amplitude=np.max(data), x_mean=x_size/2, y_mean=y_size/2,
                               x_stddev=3, y_stddev=3)
    fit_p = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:y_size, :x_size]
    p = fit_p(p_init, x, y, data)
    return p


def http_proxy_tunnel_connect(proxy, target, timeout=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(proxy)
    cmd_connect = "CONNECT %s:%d HTTP/1.1\r\n\r\n" % target
    sock.sendall(cmd_connect.encode())
    response = []
    sock.settimeout(2)  # quick hack - replace this with something better performing.
    while True:
        chunk = sock.recv(1024).decode()
        if not chunk:  # if something goes wrong
            break
        response.append(chunk)
        if "\r\n\r\n" in chunk:  # we do not want to read too far ;)
            break
    response = ''.join(response)
    return sock


def find_gauss_center(x, y, make_plot=False):
    y -= np.nanmin(y)

    def f(params):
        a = params[0]
        x0 = params[1]
        sigma = params[2]
        chi_sq = np.nansum((y - a * np.exp(-(x-x0)**2 / (2*sigma**2)))**2)
        return chi_sq
    a_guess = max(y)
    x0_guess = (x[0] + x[-1]) / 2
    sigma_guess = x0_guess - x[0]  # FIXME
    initials = a_guess, x0_guess, sigma_guess
    best_params = fmin(f, x0=initials, disp=False)
    if make_plot:
        plt.plot(x, y, "ko")
        plt.plot(x, best_params[0] * np.exp(-(x-best_params[1])**2 / (2*best_params[2]**2)))
        plt.show()
    return best_params[1]


class Decomposer(object):
    def __init__(self, image, mask, psf, gain, readnoise):
        self.no_gui = False
        # Load data
        self.data_file_name = image
        self.psf = psf
        hdu = fits.open(self.data_file_name)
        self.gain = gain
        self.readnoise = readnoise

        psf_2d = fits.getdata(self.psf)
        y, x = np.mgrid[0:psf_2d.shape[0],
                        0:psf_2d.shape[0]]
        fitted = fit_2dgaussian(psf_2d)
        self.psf_fitted = fitted(x, y)
        self.psf_fitted -= np.min(self.psf_fitted)
        self.psf_fitted /= np.sum(self.psf_fitted)
        psf_y_size, psf_x_size = psf_2d.shape
        psf_x_cen = psf_x_size // 2
        psf_y_cen = psf_y_size // 2
        self.psf = (psf_2d[psf_x_cen, :] + psf_2d[:, psf_y_cen]) / 2
        self.psf = zoom(self.psf, image_scale)
        self.psf /= np.sum(self.psf)

        self.image_data = hdu[0].data
        self.image_data[np.isnan(self.image_data)] = 0
        self.magzpt = 22.5
        self.lim_mag = 26

        # Make mask
        tmp_mask = fits.getdata(mask)
        self.mask_data = np.zeros_like(tmp_mask)
        self.mask_data[tmp_mask == 0] = 1

        # Determine pixel coordinates by WCS
        self.y_size, self.x_size = self.image_data.shape
        self.x_cen = self.x_size // 2
        self.y_cen = self.y_size // 2

        # Make new slices
        self.make_slices(fold=True)

        # Setup fitting
        self.models_list = [SimpleDiskFitter, BrokenDiskFitter, BulgeSimpleDiskFitter,
                            BulgeBrokenDiskFitter, DoubleBrokenDiskFitter,
                            BulgeDoubleBrokenDiskFitter]
        self.best_model_idx = None  # Best model by the BIC value
        self.saved_fits = {}
        self.saved_surf_bris = {}
        self.horizonthal_fitter = None
        self.vertical_fitter = None
        self.add_vertical_fitters = []
        self.selected_model_idx = -1
        self.ref_points = []
        self.requesting = False
        self.request_question = ""

        # Initialize plot
        self.messages = []
        if self.no_gui is False:
            self.fig = plt.figure(figsize=(18, 8))
            gs = self.fig.add_gridspec(nrows=3, ncols=4)
            # Original galaxy image
            self.ax_orig = self.fig.add_subplot(gs[0, 0])
            self.ax_orig.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            self.ax_orig.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
            vmin = np.percentile(self.image_data, 5)
            vmax = np.percentile(self.image_data, 99)
            self.ax_orig.imshow(self.image_data, vmin=vmin, vmax=vmax, origin="lower")
            self.center_plot_instance = self.ax_orig.plot([self.x_cen],
                                                          [self.y_cen], "kx")[0]
            self.mask_plot_instance = None
            # Horizonthal slice
            self.ax_horizonthal = self.fig.add_subplot(gs[1, 0])
            self.ax_horizonthal.set_ylim([np.nanmin(self.horizonthal_slice)-0.5, self.lim_mag])
            self.horizonthal_slice_plot_instance = None
            self.ax_horizonthal.invert_yaxis()
            self.ax_horizonthal.set_xlabel("r ['']")
            self.horizonthal_fit_plot_instances = []
            self.ref_points_plot_instances = []

            # vertical slice
            self.ax_vertical = self.fig.add_subplot(gs[2, 0])
            self.vertical_slice_plot_instance = None
            self.vertical_fit_plot_instances = []
            self.ax_vertical.set_ylim([np.nanmin(self.vertical_slice)-0.5, self.lim_mag])
            self.ax_vertical.invert_yaxis()
            self.ax_vertical.set_xlabel("r ['']")

            # Additional vertical slices
            self.ax_add_vertical = self.fig.add_subplot(gs[0:2, 1])
            self.add_vertical_slice_plot_instances = []
            self.add_vertical_fit_plot_instances = []
            self.ax_add_vertical.set_ylim([np.nanmin(self.vertical_slice)-8, self.lim_mag])
            self.ax_add_vertical.invert_yaxis()
            self.ax_add_vertical.set_xlabel("r ['']")

            # 2D model
            # self.ax_2dmodel = self.fig.add_subplot(gs[3, 0])
            # self.ax_2dmodel.set_xticks([])
            # self.ax_2dmodel.set_yticks([])
            # self.ax_2dmodel.imshow(np.zeros_like(self.image_data))

            # # 2D residual
            # self.ax_2ddiff = self.fig.add_subplot(gs[4, 0])
            # self.ax_2ddiff.set_xticks([])
            # self.ax_2ddiff.set_yticks([])
            # self.ax_2ddiff.imshow(np.zeros_like(self.image_data))

            # Help panel
            self.ax_help = self.fig.add_subplot(gs[:, 2:])
            self.ax_help.set_xticks([])
            self.ax_help.set_yticks([])
            self.models_plot_instances = []
            self.fitres_plot_instances = []
            self.requests = None
            self.request_plot_instance = None
            self.messages_plot_instances = []
            # Add permanent help info
            self.ax_help.text(x=0.05, y=0.10, s="f: fit current model using reference points",
                              transform=self.ax_help.transAxes, fontsize=12)
            self.ax_help.text(x=0.5, y=0.10, s="c: clear current fit results",
                              transform=self.ax_help.transAxes, fontsize=12)
            self.ax_help.text(x=0.5, y=0.07, s="a: auto fit current model",
                              transform=self.ax_help.transAxes, fontsize=12)
            self.ax_help.text(x=0.05, y=0.07, s="d: auto fit all models",
                              transform=self.ax_help.transAxes, fontsize=12)
            self.ax_help.text(x=0.05, y=0.04, s="s: save results on the server",
                              transform=self.ax_help.transAxes, fontsize=12)
            self.ax_help.text(x=0.5, y=0.04, s="space: mark the model as the best one",
                              transform=self.ax_help.transAxes, fontsize=12)
            self.ax_help.text(x=0.05, y=0.01, s="i: make imfit config",
                              transform=self.ax_help.transAxes, fontsize=12)
            self.ax_help.text(x=0.5, y=0.01, s="r: refine coordinates",
                              transform=self.ax_help.transAxes, fontsize=12)
            plt.tight_layout()

            # Connect events
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        else:
            # For a launch without the user control, we want to save only the galaxy and slice pictures
            self.fig = plt.figure(figsize=(6, 4))
            gs = self.fig.add_gridspec(nrows=2, ncols=1)
            # Original galaxy image
            self.ax_orig = self.fig.add_subplot(gs[0, 0])
            vmin = np.percentile(self.image_data, 5)
            vmax = np.percentile(self.image_data, 99)
            self.ax_orig.imshow(self.image_data, vmin=vmin, vmax=vmax, origin="lower")
            self.ax_orig.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            self.ax_orig.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
            self.center_plot_instance = self.ax_orig.plot([self.x_cen],
                                                          [self.y_cen], "kx")[0]
            self.mask_plot_instance = None

            self.ax_horizonthal = self.fig.add_subplot(gs[1, 0])
            self.ax_horizonthal.set_ylim([np.nanmin(self.horizonthal_slice)-0.5, self.lim_mag])
            self.horizonthal_slice_plot_instance = None
            self.ax_horizonthal.invert_yaxis()
            self.ax_horizonthal.set_xlabel("r ['']")
            self.horizonthal_fit_plot_instances = []
        self.make_plot()
        # fits.PrimaryHDU(data=self.image_data_rot).writeto(workdir / "rot.fits")
        # fits.PrimaryHDU(data=self.image_data).writeto(workdir / "cut.fits")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(workdir)

    def on_click(self, event):
        if (event.inaxes == self.ax_horizonthal) and (self.requesting):
            # Click on slice
            self.ref_points.append((event.xdata, event.ydata))
            if self.requests:
                self.request_question = self.requests.pop(0)
            else:
                self.request_question = None
                self.requesting = False
        if event.inaxes == self.ax_orig and event.button == 1:
            # Add region to mask
            r = self.image_data.shape[1] / 75
            xmin = max(0, int(event.xdata - r))
            xmax = min(self.image_data.shape[1], int(event.xdata + r))
            ymin = max(0, int(event.ydata - r))
            ymax = min(self.image_data.shape[0], int(event.ydata + r))
            self.mask_data[ymin:ymax, xmin:xmax] = 1
            self.make_slices()
        if event.inaxes == self.ax_orig and event.button == 3:
            # Remove region to mask
            r = self.image_data.shape[1] / 75
            xmin = max(0, int(event.xdata - r))
            xmax = min(self.image_data.shape[1], int(event.xdata + r))
            ymin = max(0, int(event.ydata - r))
            ymax = min(self.image_data.shape[0], int(event.ydata + r))
            self.mask_data[ymin:ymax, xmin:xmax] = 0
            self.make_slices()
        self.make_plot()

    def on_press(self, event):
        try:
            idx = int(event.key)
            self.change_model(idx)
            self.make_plot()
        except ValueError:
            pass
        if event.key == "f":
            self.fit_slices()
        if event.key == "s":
            self.save_results()
        if event.key == "a":
            self.fit_slices(auto=True)
        if event.key == "d":
            self.fit_all_models()
        if event.key == " ":
            # Mark current model as best
            self.best_model_idx = self.selected_model_idx
            self.make_plot()
        if event.key == "i":
            self.prepare_imfit_run()
            self.messages.append("Imfit config created")
            self.make_plot()
        if event.key == "r":
            self.make_slices(fold=False)
            self.refine_center()
            self.make_slices(fold=True)
            self.make_plot()
        if event.key == "c":
            self.clear()
            self.make_plot()
        if event.key == "left":
            self.x_cen -= 1
            self.make_slices(fold=True)
            self.make_plot()
        if event.key == "right":
            self.x_cen += 1
            self.make_slices(fold=True)
            self.make_plot()
        if event.key == "up":
            self.y_cen += 1
            self.make_slices(fold=True)
            self.make_plot()
        if event.key == "down":
            self.y_cen -= 1
            self.make_slices(fold=True)
            self.make_plot()

    def clear(self):
        self.horizonthal_fitter.clean()
        self.vertical_fitter.clean()
        del self.saved_fits[self.horizonthal_fitter.name.lower()]
        del self.saved_surf_bris[self.horizonthal_fitter.name.lower()]
        self.ref_points = []
        self.requests = self.models_list[self.selected_model_idx].ref_points.copy()
        self.request_question = self.requests.pop(0)
        self.requesting = True
        self.ref_points = []
        self.make_plot()

    def change_model(self, new_model_idx):
        if new_model_idx >= len(self.models_list):
            return
        self.selected_model_idx = new_model_idx
        self.horizonthal_fitter = self.models_list[new_model_idx](self.horizonthal_distances,
                                                                  self.horizonthal_slice,
                                                                  self.horizonthal_std,
                                                                  self.psf)
        if "bulge" not in self.horizonthal_fitter.name.lower():
            self.vertical_fitter = VerticalDiskFitter(self.vertical_distances,
                                                      self.vertical_slice,
                                                      self.vertical_std,
                                                      self.psf)
        else:
            self.vertical_fitter = BulgeVerticalDiskFitter(self.vertical_distances,
                                                           self.vertical_slice,
                                                           self.vertical_std,
                                                           self.psf)

        self.add_vertical_fitters = []
        for idx in range(len(self.add_vertical_slice)):
            if len(self.add_vertical_distances[idx]) > 4:
                self.add_vertical_fitters.append(AddVerticalDiskFitter(self.add_vertical_distances[idx],
                                                                       self.add_vertical_slice[idx],
                                                                       self.add_vertical_std[idx],
                                                                       self.psf))

        self.requests = self.models_list[new_model_idx].ref_points.copy()
        self.request_question = self.requests.pop(0)
        self.requesting = True
        self.ref_points = []
        if self.horizonthal_fitter.name.lower() in self.saved_fits.keys():
            self.horizonthal_fitter.restore(self.saved_fits[self.horizonthal_fitter.name.lower()]['radial'])
            self.vertical_fitter.restore(self.saved_fits[self.horizonthal_fitter.name.lower()]['vertical'])
            for idx in range(len(self.add_vertical_fitters)):
                self.add_vertical_fitters[idx].restore(self.saved_fits[f"add_vertical_{idx}"])

    def fit_slices(self, auto=False):
        # Set initial parameters to the fitter
        if auto is False:
            self.horizonthal_fitter.compute_initials(self.ref_points)
            self.horizonthal_fitter.fit()
        else:
            self.horizonthal_fitter.fit_auto()
        self.vertical_fitter.compute_initials(self.horizonthal_fitter)
        self.vertical_fitter.fit()
        if ("bulge" in self.horizonthal_fitter.name.lower()) and (auto is True):
            # Use vertical bulge as initial for horizonthal fit
            self.horizonthal_fitter.fix_bulge_to(self.vertical_fitter.par_values["re_b"],
                                                 self.vertical_fitter.par_values["mu0_b"],
                                                 self.vertical_fitter.par_values["n_b"])
            self.horizonthal_fitter.fit()
            self.horizonthal_fitter.free_bulge()
            self.horizonthal_fitter.fit()
        for idx in range(len(self.add_vertical_fitters)):
            self.add_vertical_fitters[idx].compute_initials(self.horizonthal_fitter)
            self.add_vertical_fitters[idx].fit()
        name = self.horizonthal_fitter.name.lower()
        self.saved_fits[name] = {"radial": self.horizonthal_fitter.get_all_params(),
                                 "vertical": self.vertical_fitter.get_all_params()}
        self.saved_surf_bris[name] = {"radial": self.horizonthal_fitter.get_all_data(),
                                      "vertical": self.vertical_fitter.get_all_data()}
        for idx in range(len(self.add_vertical_fitters)):
            self.saved_fits[f"add_vertical_{idx}"] = self.add_vertical_fitters[idx].get_all_params()
        self.find_best_model()
        self.make_plot()

    def fit_all_models(self):
        # Perform fitting of all models
        for idx in range(len(self.models_list)):
            print(f"Fitting model {self.models_list[idx].name}")
            self.change_model(idx)
            self.fit_slices(auto=True)
            self.make_plot()
        self.find_best_model()
        self.change_model(self.best_model_idx)
        self.make_plot()

    def find_best_model(self):
        """
        Compare BIC values of the fitted models to find the best one
        """
        best_bic = 1e10
        best_n_pars = 0
        for idx, model in enumerate(self.models_list):
            name = model.name.lower()
            if name in self.saved_fits:
                # Fit done
                bic = self.saved_fits[name]["radial"]["BIC"]
                if bic < best_bic:
                    if model.n_free_params < best_n_pars:
                        # Model has lower (i.e. better) BIC for lower number of
                        # parameters is an absolute winner
                        self.best_model_idx = idx
                        best_bic = bic
                        best_n_pars = model.n_free_params
                    else:
                        # Model has lower BIC but with the cost of bigger number
                        # parameters
                        delta_bic = best_bic - bic
                        if delta_bic > 3.5:
                            # Suppose that 3 is big enough to consider the model as a
                            # better one
                            self.best_model_idx = idx
                            best_bic = bic
                            best_n_pars = model.n_free_params

    def make_plot(self):
        # Center
        self.center_plot_instance.remove()
        self.center_plot_instance = self.ax_orig.plot([self.x_cen],
                                                      [self.y_cen], "kx")[0]
        # Plot mask
        mask_to_show = np.zeros_like(self.image_data, dtype=float)
        mask_to_show[self.mask_data == 0] *= np.nan
        if self.mask_plot_instance is not None:
            self.mask_plot_instance.remove()
            self.mask_plot_instance = None
        self.mask_plot_instance = self.ax_orig.matshow(mask_to_show, alpha=0.45, cmap="autumn", vmin=0,
                                                       vmax=2, origin='lower')

        # plot horizonthal slice
        if self.horizonthal_slice_plot_instance is not None:
            self.horizonthal_slice_plot_instance.remove()
        self.horizonthal_slice_plot_instance = None
        p = self.ax_horizonthal.errorbar(x=self.horizonthal_distances, y=self.horizonthal_slice,
                                         yerr=self.horizonthal_std, fmt="ro")
        self.horizonthal_slice_plot_instance = p
        self.ax_horizonthal.set_xlim([self.horizonthal_distances[0],
                                      self.horizonthal_distances[-1]])

        # plot vertical slice
        if self.no_gui is False:
            if self.vertical_slice_plot_instance is not None:
                self.vertical_slice_plot_instance.remove()
            self.vertical_slice_plot_instance = None
            p = self.ax_vertical.errorbar(x=self.vertical_distances, y=self.vertical_slice,
                                          yerr=self.vertical_std, fmt="ro")
            self.vertical_slice_plot_instance = p

        # Plot horizonthal fit
        while self.horizonthal_fit_plot_instances:
            self.horizonthal_fit_plot_instances.pop().remove()
        if self.horizonthal_fitter is not None:
            if not self.horizonthal_fitter.is_complex:
                # One component: just show it
                horizonthal_values = self.horizonthal_fitter.evaluate()
                if horizonthal_values is not None:
                    p = self.ax_horizonthal.plot(self.horizonthal_distances, horizonthal_values, zorder=3, color="k")
                    self.horizonthal_fit_plot_instances.append(p[0])
            else:
                horizonthal_values = self.horizonthal_fitter.evaluate(unpack=True)
                if horizonthal_values is not None:
                    for idx, v in enumerate(horizonthal_values):
                        p = self.ax_horizonthal.plot(self.horizonthal_distances, v, zorder=3, color=colors[idx])
                        self.horizonthal_fit_plot_instances.append(p[0])

        # Requests for points
        if self.no_gui is False:
            if self.request_plot_instance is not None:
                self.request_plot_instance.remove()
                self.request_plot_instance = None
            if self.requesting:
                if self.request_question:
                    self.request_plot_instance = self.ax_help.text(x=0.05, y=0.91-len(self.models_list)*0.03,
                                                                   s=f"Select: {self.request_question}",
                                                                   transform=self.ax_help.transAxes,
                                                                   fontsize=14)

        # Plot vertical fit
        if self.no_gui is False:
            while self.vertical_fit_plot_instances:
                self.vertical_fit_plot_instances.pop().remove()
            if self.vertical_fitter is not None:
                if not self.vertical_fitter.is_complex:
                    # One component: just show it
                    vertical_values = self.vertical_fitter.evaluate()
                    if vertical_values is not None:
                        p = self.ax_vertical.plot(self.vertical_distances, vertical_values, zorder=3, color="k")
                        self.vertical_fit_plot_instances.append(p[0])
                else:
                    vertical_values = self.vertical_fitter.evaluate(unpack=True)
                    if vertical_values is not None:
                        for idx, v in enumerate(vertical_values):
                            p = self.ax_vertical.plot(self.vertical_distances, v, zorder=3, color=colors[idx])
                            self.vertical_fit_plot_instances.append(p[0])

        # Plot additional vertical fits
        if self.no_gui is False:
            while self.add_vertical_slice_plot_instances:
                self.add_vertical_slice_plot_instances.pop().remove()
            for idx in range(len(self.add_vertical_std)):
                p = self.ax_add_vertical.errorbar(x=self.add_vertical_distances[idx], y=self.add_vertical_slice[idx]-2*idx,
                                                  yerr=self.add_vertical_std[idx], fmt="o", color=["r", "b"][idx % 2])
                self.add_vertical_slice_plot_instances.append(p)

        # Plot additional vertical fit
        if self.no_gui is False:
            while self.add_vertical_fit_plot_instances:
                self.add_vertical_fit_plot_instances.pop().remove()
            for idx in range(len(self.add_vertical_fitters)):
                vertical_values = self.add_vertical_fitters[idx].evaluate()
                if vertical_values is None:
                    continue
                if len(vertical_values) != len(self.add_vertical_distances[idx]):
                    continue
                if vertical_values is not None:
                    p = self.ax_add_vertical.plot(self.add_vertical_distances[idx], vertical_values-2*idx,
                                                  color="k", zorder=3)
                    self.add_vertical_fit_plot_instances.append(p[0])

        # Plot help
        # models
        if self.no_gui is False:
            while self.models_plot_instances:
                self.models_plot_instances.pop().remove()
            self.models_plot_instances.append(self.ax_help.text(x=0.05, y=0.95, s="Available models",
                                                                transform=self.ax_help.transAxes,
                                                                fontsize=14))
            for idx, model in enumerate(self.models_list):
                if idx == self.selected_model_idx:
                    color = "g"
                else:
                    color = "k"
                if model.name.lower() in self.saved_fits.keys():
                    s = f"{idx}: {model.name} (BIC: {self.saved_fits[model.name.lower()]['radial']['BIC']: 1.2f})"
                else:
                    s = f"{idx}: {model.name}"
                if (self.best_model_idx is not None) and (idx == self.best_model_idx):
                    s += " <- Best"
                self.models_plot_instances.append(self.ax_help.text(x=0.05, y=0.925-idx*0.025, s=s,
                                                                    transform=self.ax_help.transAxes,
                                                                    fontsize=14, color=color))

        # Ref points
        if self.no_gui is False:
            while self.ref_points_plot_instances:
                self.ref_points_plot_instances.pop().remove()
            for p in self.ref_points:
                self.ref_points_plot_instances.append(self.ax_horizonthal.scatter(p[0], p[1], zorder=3,
                                                                                  color="g", marker="P", s=50))

        # Fit results
        if self.no_gui is False:
            while self.fitres_plot_instances:
                self.fitres_plot_instances.pop().remove()
            if (self.horizonthal_fitter is not None) and (self.horizonthal_fitter.fit_done is True):
                fitted_params = self.horizonthal_fitter.get_all_params()
                p = self.ax_help.text(x=0.05, y=0.85-len(self.models_list)*0.025, s="Fit results:",
                                      transform=self.ax_help.transAxes, fontsize=14)
                self.fitres_plot_instances.append(p)
                p = self.ax_help.text(x=0.05, y=0.85-len(self.models_list)*0.025-0.03, s="Radial:",
                                      transform=self.ax_help.transAxes, fontsize=14)
                self.fitres_plot_instances.append(p)
                idx = 1
                for name, value in fitted_params.items():
                    if value is None:
                        continue
                    y = 0.85-idx*0.03-len(self.models_list)*0.03
                    self.fitres_plot_instances.append(self.ax_help.text(x=0.075, y=y, s=f"{name}: {value:1.2f}",
                                                                        transform=self.ax_help.transAxes, fontsize=14))
                    if "break" in name:
                        y = self.horizonthal_slice[np.argmin(np.abs(self.horizonthal_distances-value))]
                        p = self.ax_horizonthal.vlines(x=[value], ymin=y-0.75, ymax=y+0.75, linestyles="dashed", color="k")
                        self.fitres_plot_instances.append(p)
                    idx += 1
            if (self.vertical_fitter is not None) and (self.vertical_fitter.fit_done is True):
                p = self.ax_help.text(x=0.45, y=0.85-len(self.models_list)*0.025-0.03, s="Vertical:",
                                      transform=self.ax_help.transAxes, fontsize=14)
                self.fitres_plot_instances.append(p)
                fitted_params = self.vertical_fitter.get_all_params()
                idx = 1
                for name, value in fitted_params.items():
                    if value is None:
                        continue
                    y = 0.85-idx*0.03-len(self.models_list)*0.03
                    self.fitres_plot_instances.append(self.ax_help.text(x=0.475, y=y, s=f"{name}: {value:1.2f}",
                                                                        transform=self.ax_help.transAxes, fontsize=14))
                    idx += 1

        # Show messages
        if self.no_gui is False:
            while self.messages_plot_instances:
                self.messages_plot_instances.pop().remove()
            for idx, msg in enumerate(self.messages):
                y = 0.3-idx*0.03
                self.fitres_plot_instances.append(self.ax_help.text(x=0.375, y=y, s=msg, transform=self.ax_help.transAxes,
                                                                    fontsize=14))
            self.messages = []

        if self.no_gui is False:
            plt.draw()
        else:
            plt.tight_layout()
            plt.savefig(workdir / "horizonthal.png")

    def make_slices(self, width=3, fold=True):
        # Make horizonthal slice
        masked_data = self.image_data.copy()
        masked_data[self.mask_data != 0] *= np.nan
        length = min(self.x_cen, masked_data.shape[1]-self.x_cen)
        horizonthal_data = masked_data[self.y_cen-width: self.y_cen+width,
                                       self.x_cen-length: self.x_cen+length]
        horizonthal_slice = np.median(horizonthal_data, axis=0)
        horizonthal_std = np.std(horizonthal_data, axis=0)
        if fold is True:
            if len(horizonthal_slice) % 2 != 0:
                # Drop last point to make number of points even
                horizonthal_slice = horizonthal_slice[:-1]
                horizonthal_std = horizonthal_std[:-1]
            n = len(horizonthal_slice) // 2
            self.horizonthal_distances = np.empty(n)
            self.horizonthal_slice = np.empty(n)
            self.horizonthal_std = np.empty(n)
            for i in range(n):
                self.horizonthal_distances[i] = image_scale * i
                right = horizonthal_slice[n+i]
                left = horizonthal_slice[n-i]
                right_std = horizonthal_std[n+i]
                left_std = horizonthal_std[n-i]
                if (np.isnan(right)) and (np.isnan(left)):
                    self.horizonthal_slice[i] = (left+right) / 2
                    self.horizonthal_std[i] = (left_std**2 + right_std**2) ** 0.5
                elif np.isnan(right):
                    self.horizonthal_slice[i] = left
                    self.horizonthal_std[i] = left_std
                else:
                    self.horizonthal_slice[i] = right
                    self.horizonthal_std[i] = right_std
            # Remove faint fluxes
            too_faint = np.where(self.horizonthal_slice < (image_scale**2 * 10**(0.4*(self.magzpt-self.lim_mag))))
            if len(too_faint[0] > 1):
                self.horizonthal_slice = self.horizonthal_slice[:too_faint[0][0]]
                self.horizonthal_std = self.horizonthal_std[:too_faint[0][0]]
                self.horizonthal_distances = self.horizonthal_distances[:too_faint[0][0]]
        else:
            self.horizonthal_distances = image_scale * np.arange(len(horizonthal_slice))
            self.horizonthal_distances -= 0.5 * image_scale * len(horizonthal_slice)
            self.horizonthal_slice = horizonthal_slice
            self.horizonthal_std = horizonthal_std
        # Convert fluxes to magnitudes
        self.horizonthal_std = 1.08 * self.horizonthal_std / self.horizonthal_slice
        self.horizonthal_slice = -2.5 * np.log10(self.horizonthal_slice / image_scale**2) + self.magzpt
        self.horizonthal_std[self.horizonthal_std > 1] = 1

        # Make vertical slice
        length = min(self.y_cen, masked_data.shape[0]-self.y_cen)
        vertical_data = masked_data[self.y_cen-length: self.y_cen+length,
                                    self.x_cen-width: self.x_cen+width]
        vertical_slice = np.median(vertical_data, axis=1)
        vertical_std = np.std(vertical_data, axis=1)
        if fold is True:
            if len(vertical_slice) % 2 != 0:
                # Drop last point to make number of points even
                vertical_slice = vertical_slice[:-1]
                vertical_std = vertical_std[:-1]
            n = len(vertical_slice) // 2
            self.vertical_distances = np.empty(n)
            self.vertical_slice = np.empty(n)
            self.vertical_std = np.empty(n)
            for i in range(n):
                self.vertical_distances[i] = image_scale * i
                right = vertical_slice[n+i]
                left = vertical_slice[n-i]
                right_std = vertical_std[n+i]
                left_std = vertical_std[n-i]
                if (np.isnan(right)) and (np.isnan(left)):
                    self.vertical_slice[i] = (left+right) / 2
                    self.vertical_std[i] = (left_std**2 + right_std**2) ** 0.5
                elif np.isnan(right):
                    self.vertical_slice[i] = left
                    self.vertical_std[i] = left_std
                else:
                    self.vertical_slice[i] = right
                    self.vertical_std[i] = right_std
            # Remove faint fluxes
            too_faint = np.where(self.vertical_slice < (image_scale**2 * 10**(0.4*(self.magzpt-self.lim_mag))))
            if len(too_faint[0] > 1):
                self.vertical_slice = self.vertical_slice[:too_faint[0][0]]
                self.vertical_std = self.vertical_std[:too_faint[0][0]]
                self.vertical_distances = self.vertical_distances[:too_faint[0][0]]
        else:
            self.vertical_distances = image_scale * np.arange(len(vertical_slice))
            self.vertical_distances -= 0.5 * image_scale * len(vertical_slice)
            self.vertical_slice = vertical_slice
            self.vertical_std = vertical_std
        # Convert fluxes to magnitudes
        self.vertical_std = 1.08 * self.vertical_std / self.vertical_slice
        self.vertical_std[self.vertical_std > 1] = 1
        self.vertical_slice = -2.5 * np.log10(self.vertical_slice / image_scale**2) + self.magzpt

        # Make additional vertical slices
        rmax = max(self.horizonthal_distances / image_scale)
        shifts = [int(-0.8*rmax), int(-0.65*rmax), int(-0.5*rmax),
                  int(0.5*rmax), int(0.65*rmax), int(0.8*rmax)]
        self.add_vertical_distances = []
        self.add_vertical_slice = []
        self.add_vertical_std = []
        for shift in shifts:
            vertical_data = masked_data[self.y_cen-length: self.y_cen+length,
                                        self.x_cen-width+shift: self.x_cen+width+shift]
            flux = np.median(vertical_data, axis=1)
            flux_std = np.std(vertical_data, axis=1)
            distances = image_scale * np.arange(len(vertical_slice))
            distances -= 0.5 * image_scale * len(vertical_slice)
            mag = -2.5 * np.log10(flux / image_scale**2) + self.magzpt
            mag_std = 1.08*flux_std/flux
            mag_std[mag_std > 1] = 1
            distances = distances[np.isfinite(mag)]
            mag_std = mag_std[np.isfinite(mag)]
            mag = mag[np.isfinite(mag)]
            # Drop faint wings
            center = len(mag)//2
            for i_lim in range(center):
                if (mag[i_lim+center] > self.lim_mag) and (mag[i_lim-center] > self.lim_mag):
                    break
            if len(mag) > 5:
                self.add_vertical_distances.append(distances[center-i_lim:center+i_lim])
                self.add_vertical_slice.append(mag[center-i_lim:center+i_lim])
                self.add_vertical_std.append(mag_std[center-i_lim:center+i_lim])

    def refine_center(self):
        max_idx = np.nanargmin(self.horizonthal_slice)
        width = 7  # len(self.horizonthal_distances) // 10
        central_region_dists = self.horizonthal_distances[max_idx-width: max_idx+width]
        central_region_surfbri = self.horizonthal_slice[max_idx-width: max_idx+width]
        horizonthal_center = find_gauss_center(central_region_dists, -central_region_surfbri)
        max_idx = np.nanargmin(self.vertical_slice)
        width = 7  # len(self.vertical_distances) // 10
        central_region_dists = self.vertical_distances[max_idx-width: max_idx+width]
        central_region_surfbri = self.vertical_slice[max_idx-width: max_idx+width]
        vertical_center = find_gauss_center(central_region_dists, -central_region_surfbri)
        horizonthal_shift = horizonthal_center / image_scale
        vertical_shift = vertical_center / image_scale
        self.x_cen = int(self.x_cen + horizonthal_shift)
        self.y_cen = int(self.y_cen + vertical_shift)

    def save_results(self):
        all_results = {}
        # Save slices
        all_results["radial_distances['']"] = list(self.horizonthal_distances.astype(float))
        all_results["radial_surf_bri"] = list(self.horizonthal_slice.astype(float))
        all_results["radial_surf_std"] = list(self.horizonthal_std.astype(float))
        all_results["vertical_distances['']"] = list(self.vertical_distances.astype(float))
        all_results["vertical_surf_bri"] = list(self.vertical_slice.astype(float))
        all_results["vertical_surf_std"] = list(self.vertical_std.astype(float))
        # Save fit params
        # Coordinates of the center and position angle
        all_results["x_cen"] = self.x_cen
        all_results["y_cen"] = self.y_cen
        all_results["posang"] = self.posang
        # Cutout region
        all_results["x_min"] = self.x_min
        all_results["x_max"] = self.x_max
        all_results["y_min"] = self.y_min
        all_results["x_max"] = self.x_max
        # 1) A dictionary for all fit params
        all_results["fit_params"] = self.saved_fits
        # 2) A dictionary for all fitted surface brightnesses
        all_results["surf_brightnesses"] = self.saved_surf_bris
        # 3) Name of the best model
        all_results["best_model"] = self.models_list[self.best_model_idx].name.lower()
        # Save json
        filename = "Slice_fit.json"
        fout = open(workdir / filename, "w")
        json.dump(all_results, fout)
        fout.close()
        # Save image for decomposition
        self.prepare_imfit_run()
        self.make_plot()

    def prepare_imfit_run(self):
        """
        Make a directory with imfit run
        """
        self.imfit_run_path = Path("imfit_run")
        if not self.imfit_run_path.exists():
            os.makedirs(self.imfit_run_path)
        fits.PrimaryHDU(data=self.psf_fitted).writeto(self.imfit_run_path / "psf.fits", overwrite=True)
        header = fits.Header({"MAGZPT": self.magzpt})
        fits.PrimaryHDU(data=self.image_data, header=header).writeto(self.imfit_run_path / "image.fits",
                                                                     overwrite=True)
        fits.PrimaryHDU(data=self.mask_data).writeto(self.imfit_run_path / "mask.fits", overwrite=True)
        imfit_config = open(self.imfit_run_path / "config.imfit", "w")
        if self.gain is not None:
            imfit_config.write(f"GAIN {self.gain}\n")
            imfit_config.write(f"READNOISE {self.readnoise}\n")
        try:
            z0 = np.nanmedian([vf.par_values["z0_d"] for vf in self.add_vertical_fitters])
        except TypeError:
            z0 = 3
        func_part = self.horizonthal_fitter.to_imfit(self.x_cen, self.y_cen, self.magzpt, image_scale)
        pars = {}
        pars["vertical_scale"] = z0
        if "bulge" in self.horizonthal_fitter.name.lower():
            ell = 1 - self.vertical_fitter.par_values["re_b"] / self.horizonthal_fitter.par_values["re_b"]
            if ell > 0.5:
                ell = 0.5
            if ell < 0.05:
                ell = 0.05
            pars["bulge_ell"] = ell
        func_part = Template(func_part).substitute(pars)
        imfit_config.write(func_part)
        imfit_config.close()


def setup():
    if workdir.exists():
        shutil.rmtree(workdir)
    os.makedirs(workdir)


def main(args):
    setup()
    # Manual intervention, no remote results
    with Decomposer(args.image, args.mask, args.psf, args.gain, args.rnoise) as d:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="FITS file with image")
    parser.add_argument("--mask", type=Path, help="FITS file with mask")
    parser.add_argument("--psf", type=Path, help="FITS file with PSF")
    parser.add_argument("--gain", type=float, help="Gain value", default=None)
    parser.add_argument("--rnoise", type=float, help="Read noise value", default=None)
    args = parser.parse_args()
    main(args)
