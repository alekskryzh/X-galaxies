import math
import numpy as np
from scipy.optimize import fmin
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.integrate import simps

kappa = 2.5 / math.log(10)


def bic(y_data, y_err, y_fit, n_params):
    """
    Bayessian information criterion
    """
    loglike = np.sum(norm.logpdf(y_fit, loc=y_data, scale=y_err))
    bic = n_params * np.log(len(y_data)) - 2 * loglike
    return bic


class BasicFitter(object):
    def __init__(self, radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free):
        self.is_complex = False
        self.radii = radii
        self.surf_bri = surf_bri
        self.surf_bri_std = surf_bri_std
        if (np.sum(psf) != 0) and (not np.isnan(np.sum(psf))):
            self.psf = psf
        else:
            self.psf = np.zeros(3)
            self.psf[1] = 1
        self.fit_done = False
        self.chi = None
        self.bic = None
        self.par_names = par_names
        self.par_dims = par_dims
        self.par_free = {name: free for name, free in zip(par_names, par_free)}
        self.par_values = {name: None for name in par_names}

    def compute_non_free(self):
        pass

    def set_params(self, new_values):
        idx = 0
        for name in self.par_names:
            if self.par_free[name]:
                self.par_values[name] = new_values[idx]
                idx += 1
        self.compute_non_free()

    def get_all_params(self):
        params = {}
        # Load values of fitted params
        for idx in range(len(self.par_names)):
            name = self.par_names[idx]
            dim = self.par_dims[idx]
            if dim != "":
                key = f"{name}[{dim}]"
            else:
                key = name
            params[key] = self.par_values[name]
        # Add BIC and Chi squared values
        params["BIC"] = self.bic
        params["ChiSq"] = self.chi
        params["BIC_clean"] = self.bic_clean
        return params

    def clean(self):
        self.fit_done = False
        self.chi = None
        self.bic = None
        self.par_values = {name: None for name in self.par_names}

    def get_all_data(self):
        """
        Returns all surf brightnesses fits
        """
        surf_bri = {}
        values = self.evaluate(True)
        if self.is_complex:
            for i in (0, 1, 2):
                values[i][np.isnan(values[i])] = 99.9
            surf_bri["is_complex"] = True
            surf_bri["surf_brightnesses"] = {"bulge": list(values[0].astype(float)),
                                             "disk": list(values[1].astype(float)),
                                             "total": list(values[2].astype(float))}
        else:
            values[np.isnan(values)] = 99.9
            surf_bri["is_complex"] = False
            surf_bri["surf_brightnesses"] = list(values.astype(float))
        return surf_bri

    def get_par_values(self):
        values = []
        for name in self.par_names:
            if self.par_free[name]:
                values.append(self.par_values[name])
        return values

    def restore(self, loaded_params):
        """
        Restore saved condition
        """
        for idx in range(len(self.par_names)):
            name = self.par_names[idx]
            dim = self.par_dims[idx]
            if dim != "":
                key = f"{name}[{dim}]"
            else:
                key = name
            self.par_values[name] = loaded_params[key]
        self.bic = loaded_params["BIC"]
        self.chi = loaded_params["ChiSq"]
        self.bic_clean = loaded_params["BIC_clean"]
        self.fit_done = True

    def fit(self):
        def chi_sq(params):
            self.set_params(params)
            self.chi = np.sum(((self.surf_bri - self.evaluate()) / self.surf_bri_std)**2)
            self.chi *= self.chi_penalty()
            return self.chi
        fmin(chi_sq, x0=self.get_par_values(), disp=False)
        self.fit_done = True
        penalty = self.bic_penalty()
        self.bic = bic(self.surf_bri, self.surf_bri_std, self.evaluate(), 2) + penalty
        self.bic_clean = self.bic - penalty

    def to_imfit(self, *args):
        """
        Create an imfit function config for the fit results
        """
        return ""

    def bic_penalty(self):
        """
        Additinal penalty to BIC
        """
        return 0

    def chi_penalty(self):
        """
        Multiplicative penalty to chi squared for bad parameters
        """
        return 1


class SimpleDiskFitter(BasicFitter):
    name = "Simple disk"
    ref_points = ["First point on disk", "second point on disk"]
    n_free_params = 2

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        par_names = ("h_d", "mu0_d")
        par_dims = ("''", "mag/sq.''")
        par_free = (True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        m0 = 25
        mag = self.par_values["mu0_d"] + kappa * self.radii / self.par_values["h_d"]
        flux_orig = 10 ** (0.4 * (m0 - mag))
        flux_orig_symmetric = np.append(flux_orig[::-1], flux_orig)  # Make symmetric for the padding purposes
        flux_convolved = fftconvolve(flux_orig_symmetric, self.psf, mode="same")[len(flux_orig):]
        return -2.5 * np.log10(flux_convolved) + m0

    def compute_initials(self, ref_points):
        h_d = kappa * (ref_points[0][0] - ref_points[1][0]) / (ref_points[0][1]-ref_points[1][1])
        mu0_d = ref_points[0][1] - kappa * ref_points[0][0] / h_d
        self.set_params([h_d, mu0_d])

    def fit_auto(self):
        """
        Try to determine good initial parameters to and perform fit
        """
        mu0_d = self.surf_bri[0]
        h_d = kappa*((self.radii[-1] - self.radii[0]) / (self.surf_bri[-1] - self.surf_bri[0]))
        self.set_params([h_d, mu0_d])
        self.fit()

    def get_inner_disk_mu0(self):
        return self.par_values["mu0_d"]

    def to_imfit(self, x_cen, y_cen, magzpt, image_scale):
        imfit = f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION ExponentialDisk3D\n"
        imfit += "PA  90  80,100\n"
        imfit += "inc  90 fixed\n"
        j0 = 10**(0.4 * (magzpt - self.par_values["mu0_d"])) * image_scale**2 / (2*self.par_values["h_d"])
        imfit += f"J_0 {j0}\n"
        h = self.par_values['h_d']/image_scale
        if h < 0:
            h = 100
        imfit += f"h {h}\n"
        imfit += "n 2 0.5,20\n"
        imfit += "z_0 $vertical_scale \n"
        return imfit


class BulgeSimpleDiskFitter(BasicFitter):
    name = "Bulge + simple disk"
    ref_points = ["First point on bulge", "Second point on bulge", "Third point on bulge",
                  "First point on disk", "Second point on disk"]
    n_free_params = 5

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        par_names = ("re_b", "mu0_b", "n_b", "h_d", "mu0_d")
        par_dims = ("''", "mag/sq.''", "", "''", "mag/sq.''")
        par_free = (True, True, True, True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)
        self.old_bulge_re = None
        self.is_complex = True

    def fix_bulge_to(self, re_b, mu0_b, n_b):
        self.par_values["re_b"] = re_b
        self.par_values["mu0_b"] = mu0_b
        self.par_values["n_b"] = n_b
        self.par_free["re_b"] = False
        self.par_free["mu0_b"] = False
        self.par_free["n_b"] = False

    def free_bulge(self):
        self.old_bulge_re = self.par_values["re_b"]
        self.par_free["re_b"] = True
        self.par_free["mu0_b"] = True
        self.par_free["n_b"] = True

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        m0 = 25
        nu_b = 2 * self.par_values["n_b"] - 1/3 + 4 / (405*self.par_values["n_b"])
        disk = 10**(0.4 * (m0 - (self.par_values["mu0_d"] + kappa * self.radii / self.par_values["h_d"])))
        bulge = 10**(0.4 * (m0 - (self.par_values["mu0_b"] +
                                  kappa * nu_b * (self.radii/self.par_values["re_b"]) ** (1/self.par_values["n_b"]))))
        week_bulge_idx = np.where(bulge < 0.01)[0]
        if len(week_bulge_idx):
            bulge[week_bulge_idx[0]:] = 0.01
        disk_flux = simps(disk*self.radii, self.radii)
        bulge_flux = simps(bulge*self.radii, self.radii)
        self.bulge_to_total = bulge_flux / (bulge_flux + disk_flux)
        total = disk + bulge
        if unpack is False:
            # Perform convolution of the total flux
            total_symmetric = np.append(total[::-1], total)
            total_convolved = fftconvolve(total_symmetric, self.psf, mode="same")[len(total):]
            return -2.5 * np.log10(total_convolved) + m0
        else:
            bulge_symmetric = np.append(bulge[::-1], bulge)
            bulge_convolved = fftconvolve(bulge_symmetric, self.psf, mode="same")[len(bulge):]
            disk_symmetric = np.append(disk[::-1], disk)
            disk_convolved = fftconvolve(disk_symmetric, self.psf, mode="same")[len(disk):]
            total_symmetric = np.append(total[::-1], total)
            total_convolved = fftconvolve(total_symmetric, self.psf, mode="same")[len(total):]
            bulge = -2.5 * np.log10(bulge_convolved) + m0
            disk = -2.5 * np.log10(disk_convolved) + m0
            total = -2.5 * np.log10(total_convolved) + m0
            return (bulge, disk, total)

    def compute_initials(self, ref_points):
        # Disk
        h_d = kappa * (ref_points[3][0] - ref_points[4][0]) / (ref_points[3][1]-ref_points[4][1])
        mu0_d = ref_points[3][1] - kappa * ref_points[3][0] / h_d
        # Bulge
        m0 = 25
        f_total = 10 ** (0.4 * (m0 - ref_points[0][1]))
        f_disk = 10 ** (0.4 * (m0 - mu0_d))
        f_bulge = f_total - f_disk
        mu0_b = -2.5 * np.log10(f_bulge) + m0
        re_b = ref_points[2][0] / 2
        k1 = (ref_points[1][1]-ref_points[0][1]) / (ref_points[1][0]-ref_points[0][0])
        k2 = (ref_points[2][1]-ref_points[1][1]) / (ref_points[2][0]-ref_points[1][0])
        if k1 < k2:
            n_b = 0.5
        else:
            n_b = 2
        self.set_params([re_b, mu0_b, n_b, h_d, mu0_d])

    def compute_non_free(self):
        self.par_values["nu_b"] = 2 * self.par_values["n_b"] - 1/3 + 4 / (405*self.par_values["n_b"])

    def fit_auto(self):
        """
        Try to determine good initial parameters to and perform fit
        """
        # Lets fit disk at first, then look for what is left and use it find bulge parameter
        # Get two points on the disk
        i1 = len(self.radii) // 2
        r1 = (self.radii[i1] + self.radii[i1-1]) / 2
        mu1 = (self.surf_bri[i1] + self.surf_bri[i1-1]) / 2
        i2 = 3*len(self.radii) // 4
        r2 = (self.radii[i2] + self.radii[i2-1]) / 2
        mu2 = (self.surf_bri[i2] + self.surf_bri[i2-1]) / 2
        # Compute h and mu0
        dr = r2 - r1
        dmu = mu2 - mu1
        hd_guess = kappa * dr / dmu
        mu0d_guess = mu1 - kappa * r1 / hd_guess
        surf_bri_disk = (mu0d_guess + kappa*self.radii/hd_guess)
        # Subtract the disk
        m0 = 25
        f_total = 10 ** (0.4 * (m0 - self.surf_bri))
        f_disk = 10 ** (0.4 * (m0 - surf_bri_disk))
        f_bulge = f_total - f_disk
        # Find bulge params
        if f_bulge[0] > 0:
            mu0_b_guess = -2.5 * np.log10(f_bulge[0]) + m0
            for i, f in enumerate(f_bulge):
                if f < 0.666 * f_bulge[0]:
                    re_b_guess = self.radii[i]
                    break
            for i, f in enumerate(f_bulge):
                if f < 0.333 * f_bulge[0]:
                    r_out = self.radii[i]
            if 2 * re_b_guess > r_out:
                nb_guess = 0.5
            else:
                nb_guess = 2.0
        else:
            re_b_guess = self.radii[1]
            mu0_b_guess = self.surf_bri[0] - 2
            nb_guess = 1.0
        self.set_params([re_b_guess, mu0_b_guess, nb_guess, hd_guess, mu0d_guess])
        self.fit()

    def get_inner_disk_mu0(self):
        return self.par_values["mu0_d"]

    def to_imfit(self, x_cen, y_cen, magzpt, image_scale):
        imfit = f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION Sersic\n"
        imfit += "PA  90  80,100\n"
        imfit += "ell $bulge_ell   0.0,0.6\n"
        n_b = self.par_values['n_b']
        if n_b < 0.5:
            n_b = 0.5
        if n_b > 6:
            n_b = 6
        imfit += f"n {n_b}   0.3,8\n"
        nu_b = 2 * self.par_values["n_b"] - 1/3 + 4 / (405*self.par_values["n_b"])
        mue = self.par_values["mu0_b"] + 2.5 * nu_b / np.log(10)
        ie = 10**(0.4*(magzpt - mue)) * image_scale**2
        imfit += f"I_e {ie}\n"
        imfit += f"r_e  {self.par_values['re_b']/image_scale}\n"
        imfit += f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION ExponentialDisk3D\n"
        imfit += "PA  90 80,100\n"
        imfit += "inc  90 fixed\n"
        j0 = 10**(0.4 * (magzpt - self.par_values["mu0_d"])) * image_scale**2 / (2*self.par_values["h_d"])
        imfit += f"J_0 {j0}\n"
        h = self.par_values['h_d']/image_scale
        if h < 0:
            h = 100
        imfit += f"h {h}\n"
        imfit += "n 2 0.5,20\n"
        imfit += "z_0 $vertical_scale\n"
        return imfit

    def chi_penalty(self):
        total = 1
        # Penalise huge bulge
        if self.bulge_to_total > 0.25:
            total *= 1+5*(self.bulge_to_total - 0.25)
        # Penalize small bulges Sersic values
        if self.par_values["n_b"] < 0.4:
            total *= 1 + 10*(0.4-self.par_values["n_b"])
        if self.old_bulge_re is not None:
            # Penalize bulge Re difference from the vertical fit
            diff = (self.par_values["re_b"] - self.old_bulge_re) / self.old_bulge_re
            if diff > 0.25:
                total *= 1 + 50*diff
        return total


class BrokenDiskFitter(BasicFitter):
    name = "Broken expdisk"
    ref_points = ["Point on first disk", "Break point", "Point on second disk"]
    n_free_params = 4

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        par_names = ("mu0_d1", "mu0_d2", "h_d1", "h_d2", "r_break")
        par_dims = ("mag/sq.''", "mag/sq.''", "''", "''", "''")
        par_free = (True, False, True, True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        mag = np.zeros_like(self.radii)
        inner = np.where(self.radii <= self.par_values["r_break"])
        mag[inner] = self.par_values["mu0_d1"] + kappa * self.radii[inner] / self.par_values["h_d1"]
        outer = np.where(self.radii > self.par_values["r_break"])
        mag[outer] = self.par_values["mu0_d2"] + kappa * self.radii[outer] / self.par_values["h_d2"]
        # Convolution
        m0 = 25
        flux_orig = 10 ** (0.4 * (m0 - mag))
        flux_orig_symmetric = np.append(flux_orig[::-1], flux_orig)  # Make symmetric for the padding purposes
        flux_convolved = fftconvolve(flux_orig_symmetric, self.psf, mode="same")[len(flux_orig):]
        return -2.5 * np.log10(flux_convolved) + m0

    def compute_initials(self, ref_points):
        h_d1 = kappa * (ref_points[0][0] - ref_points[1][0]) / (ref_points[0][1]-ref_points[1][1])
        mu0_d1 = ref_points[0][1] - kappa * ref_points[0][0] / h_d1
        h_d2 = kappa * (ref_points[1][0] - ref_points[2][0]) / (ref_points[1][1]-ref_points[2][1])
        r_break = ref_points[1][0]
        self.set_params([mu0_d1, h_d1, h_d2, r_break])

    def compute_non_free(self):
        self.par_values["mu0_d2"] = (self.par_values["mu0_d1"] +
                                     kappa * self.par_values["r_break"] * (1/self.par_values["h_d1"] -
                                                                           1/self.par_values["h_d2"]))

    def fit_auto(self):
        """
        Try to determine good initial parameters to and perform fit
        """
        mu0_d1 = self.surf_bri[0]
        r1 = len(self.radii) // 3
        r2 = 3 * len(self.radii) // 4
        step = (r2 - r1) // 10
        best_chi = 1e10
        best_params = (20, 10, 10, 30)
        for r_break_idx in range(r1, r2, step):
            h_d1 = kappa * (self.radii[0] - self.radii[r_break_idx]) / (self.surf_bri[0]-self.surf_bri[r_break_idx])
            h_d2 = kappa * (self.radii[r_break_idx] - self.radii[-1]) / (self.surf_bri[r_break_idx]-self.surf_bri[-1])
            r_break = self.radii[r_break_idx]
            self.set_params([mu0_d1, h_d1, h_d2, r_break])
            self.fit()
            if self.chi < best_chi:
                best_chi = self.chi
                best_params = (mu0_d1, h_d1, h_d2, r_break)
        self.set_params(best_params)
        self.fit()

    def get_inner_disk_mu0(self):
        return self.par_values["mu0_d1"]

    def to_imfit(self, x_cen, y_cen, magzpt, image_scale):
        imfit = f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION BknExp3D\n"
        imfit += "PA  90  80,100\n"
        imfit += "inc  90  fixed\n"
        j0 = 10**(0.4 * (magzpt - self.par_values["mu0_d1"])) * image_scale**2 / (2*self.par_values["h_d1"])
        imfit += f"J_0 {j0}\n"
        h1 = self.par_values['h_d1']/image_scale
        if h1 < 0:
            h1 = 100
        h2 = self.par_values['h_d2']/image_scale
        if h2 < 0:
            h2 = 100
        imfit += f"h1 {h1}\n"
        imfit += f"h2 {h2}\n"
        imfit += f"r_break {self.par_values['r_break']/image_scale}\n"
        imfit += "n 2  0.5,20\n"
        imfit += "z_0 $vertical_scale\n"
        return imfit

    def bic_penalty(self):
        total = 0
        # Penalise very short outer disk
        if (max(self.radii) - self.par_values["r_break"]) / max(self.radii) < 0.2:
            total += 100
        elif len(np.where(self.radii > self.par_values["r_break"])[0]) < 4:
            total += 100
        # Penalise if inner disk mimicing the bulge
        if (self.par_values["r_break"] < 0.2 * self.radii[-1]):
            total += 5
        return total

    def chi_penalty(self):
        # Penalise if inner disk is smaller than the bulge
        total = 1
        # Penalise if inner disk mimicing the bulge
        if self.par_values["r_break"] < 0.2 * self.radii[-1]:
            total *= 0.2 * self.radii[-1] / self.par_values["r_break"]
        return total


class BulgeBrokenDiskFitter(BasicFitter):
    name = "Bulge + broken disk"
    ref_points = ["First point on bulge", "Second point on bulge", "Third point on bulge",
                  "Point on first disk", "Break point", "Point on second disk"]
    n_free_params = 7

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        par_names = ("re_b", "mu0_b", "n_b", "mu0_d1", "mu0_d2", "h_d1", "h_d2", "r_break")
        par_dims = ("''", "mag/sq.''", "n_b", "mag/sq.''", "''", "mag/sq.''", "''", "''")
        par_free = (True, True, True, True, False, True, True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)
        self.old_bulge_re = None
        self.is_complex = True

    def fix_bulge_to(self, re_b, mu0_b, n_b):
        self.par_values["re_b"] = re_b
        self.par_values["mu0_b"] = mu0_b
        self.par_values["n_b"] = n_b
        self.par_free["re_b"] = False
        self.par_free["mu0_b"] = False
        self.par_free["n_b"] = False

    def free_bulge(self):
        self.old_bulge_re = self.par_values["re_b"]
        self.par_free["re_b"] = True
        self.par_free["mu0_b"] = True
        self.par_free["n_b"] = True

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        m0 = 25
        nu_b = 2 * self.par_values["n_b"] - 1/3 + 4 / (405*self.par_values["n_b"])
        disk = np.zeros_like(self.radii)
        inner = np.where(self.radii <= self.par_values["r_break"])
        disk[inner] = self.par_values["mu0_d1"] + kappa * self.radii[inner] / self.par_values["h_d1"]
        outer = np.where(self.radii > self.par_values["r_break"])
        disk[outer] = self.par_values["mu0_d2"] + kappa * self.radii[outer] / self.par_values["h_d2"]

        disk = 10**(0.4 * (m0 - disk))
        bulge = 10**(0.4 * (m0 - (self.par_values["mu0_b"] +
                                  kappa * nu_b * (self.radii/self.par_values["re_b"]) ** (1/self.par_values["n_b"]))))
        week_bulge_idx = np.where(bulge < 0.01)[0]
        if len(week_bulge_idx):
            bulge[week_bulge_idx[0]:] = 0.01
        disk_flux = simps(disk*self.radii, self.radii)
        bulge_flux = simps(bulge*self.radii, self.radii)
        self.bulge_to_total = bulge_flux / (bulge_flux + disk_flux)
        total = disk + bulge
        if unpack is False:
            # Perform convolution of the total flux
            total_symmetric = np.append(total[::-1], total)
            total_convolved = fftconvolve(total_symmetric, self.psf, mode="same")[len(total):]
            return -2.5 * np.log10(total_convolved) + m0
        else:
            bulge_symmetric = np.append(bulge[::-1], bulge)
            bulge_convolved = fftconvolve(bulge_symmetric, self.psf, mode="same")[len(bulge):]
            disk_symmetric = np.append(disk[::-1], disk)
            disk_convolved = fftconvolve(disk_symmetric, self.psf, mode="same")[len(disk):]
            total_symmetric = np.append(total[::-1], total)
            total_convolved = fftconvolve(total_symmetric, self.psf, mode="same")[len(total):]
            bulge = -2.5 * np.log10(bulge_convolved) + m0
            disk = -2.5 * np.log10(disk_convolved) + m0
            total = -2.5 * np.log10(total_convolved) + m0
            return (bulge, disk, total)

    def compute_non_free(self):
        self.par_values["mu0_d2"] = (self.par_values["mu0_d1"] +
                                     kappa * self.par_values["r_break"] * (1/self.par_values["h_d1"]-
                                                                           1/self.par_values["h_d2"]))

    def compute_initials(self, ref_points):
        # Disk
        h_d1 = kappa * (ref_points[3][0] - ref_points[4][0]) / (ref_points[3][1]-ref_points[4][1])
        mu0_d1 = ref_points[3][1] - kappa * ref_points[3][0] / h_d1
        h_d2 = kappa * (ref_points[4][0] - ref_points[5][0]) / (ref_points[4][1]-ref_points[5][1])
        r_break = ref_points[4][0]
        # Bulge
        m0 = 25
        f_total = 10 ** (0.4 * (m0 - ref_points[0][1]))
        f_disk = 10 ** (0.4 * (m0 - mu0_d1))
        f_bulge = f_total - f_disk
        mu0_b = -2.5 * np.log10(f_bulge) + m0
        re_b = ref_points[2][0] / 2
        k1 = (ref_points[1][1]-ref_points[0][1]) / (ref_points[1][0]-ref_points[0][0])
        k2 = (ref_points[2][1]-ref_points[1][1]) / (ref_points[2][0]-ref_points[1][0])
        if k1 < k2:
            n_b = 0.5
        else:
            n_b = 2

        self.set_params([re_b, mu0_b, n_b, mu0_d1, h_d1, h_d2, r_break])

    def fit_auto(self):
        """
        Try to determine good initial parameters to and perform fit
        """
        # Suppose the bulge dominates the disk in the inner quareter of the radii
        # and everything outside of the inner quarter is disk. So let's fit the
        # outer two quarters with the broken disk and then fit the bulge to what
        # was left
        disk_start_idx = len(self.radii) // 4
        mu0_d1 = self.surf_bri[disk_start_idx]
        r_break_min_idx = len(self.radii) // 3
        r_break_max_idx = 3 * len(self.radii) // 4
        step = (r_break_max_idx - r_break_min_idx) // 10
        best_chi = 1e10

        for r_break_idx in range(r_break_min_idx, r_break_max_idx, step):
            h_d1 = kappa * ((self.radii[disk_start_idx] - self.radii[r_break_idx]) /
                            (self.surf_bri[disk_start_idx]-self.surf_bri[r_break_idx]))
            h_d2 = kappa * (self.radii[r_break_idx] - self.radii[-1]) / (self.surf_bri[r_break_idx]-self.surf_bri[-1])
            r_break = self.radii[r_break_idx]
            surf_bri_disk = (mu0_d1 + kappa*self.radii/h_d1)
            # Subtract the disk
            m0 = 25
            f_total = 10 ** (0.4 * (m0 - self.surf_bri))
            f_disk = 10 ** (0.4 * (m0 - surf_bri_disk))
            f_bulge = f_total - f_disk
            # Find bulge params
            if f_bulge[0] > 0:
                mu0_b_guess = -2.5 * np.log10(f_bulge[0]) + m0
                for i, f in enumerate(f_bulge):
                    if f < 0.666 * f_bulge[0]:
                        re_b_guess = self.radii[i]
                        break
                for i, f in enumerate(f_bulge):
                    if f < 0.333 * f_bulge[0]:
                        r_out = self.radii[i]
                if 2 * re_b_guess > r_out:
                    nb_guess = 0.5
                else:
                    nb_guess = 2.0
            else:
                re_b_guess = self.radii[1]
                mu0_b_guess = self.surf_bri[0] - 2
                nb_guess = 1.0
            self.set_params([re_b_guess, mu0_b_guess, nb_guess, mu0_d1, h_d1, h_d2, r_break])
            self.fit()
            if self.chi < best_chi:
                best_chi = self.chi
                best_params = (re_b_guess, mu0_b_guess, nb_guess,
                               mu0_d1, h_d1, h_d2, r_break)
        self.set_params(best_params)
        self.fit()

    def get_inner_disk_mu0(self):
        return self.par_values["mu0_d1"]

    def to_imfit(self, x_cen, y_cen, magzpt, image_scale):
        nu_b = 2 * self.par_values["n_b"] - 1/3 + 4 / (405*self.par_values["n_b"])
        imfit = f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION Sersic\n"
        imfit += "PA  90 80,100\n"
        imfit += "ell  $bulge_ell   0.0,0.6\n"
        n_b = self.par_values['n_b']
        if n_b < 0.5:
            n_b = 0.5
        if n_b > 6:
            n_b = 6
        imfit += f"n {n_b}   0.3,8\n"
        mue = self.par_values["mu0_b"] + 2.5 * nu_b / np.log(10)
        ie = 10**(0.4*(magzpt - mue)) * image_scale**2
        imfit += f"I_e {ie}\n"
        imfit += f"r_e  {self.par_values['re_b']/image_scale}\n"
        imfit += f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION BknExp3D\n"
        imfit += "PA  90  80,100\n"
        imfit += "inc  90 fixed\n"
        j0 = 10**(0.4 * (magzpt - self.par_values["mu0_d1"])) * image_scale**2 / (2*self.par_values["h_d1"])
        imfit += f"J_0 {j0}\n"
        h1 = self.par_values['h_d1']/image_scale
        if h1 < 0:
            h1 = 100
        h2 = self.par_values['h_d2']/image_scale
        if h2 < 0:
            h2 = 100
        imfit += f"h1 {h1}\n"
        imfit += f"h2 {h2}\n"
        imfit += f"r_break {self.par_values['r_break']/image_scale}\n"
        imfit += "n 2 0.5,20\n"
        imfit += "z_0 $vertical_scale \n"
        return imfit

    def bic_penalty(self):
        """
        Penalise very short outer disk
        """
        if (max(self.radii) - self.par_values["r_break"]) / max(self.radii) < 0.2:
            return 100
        elif len(np.where(self.radii > self.par_values["r_break"])[0]) < 4:
            return 100
        return 0

    def chi_penalty(self):
        # Penalise if inner disk is smaller than the bulge
        total = 1
        if self.par_values["r_break"] < self.par_values["re_b"]:
            total *= abs(self.par_values["re_b"] / self.par_values["r_break"])
        # Penalise if inner disk mimicing the bulge
        if self.par_values["r_break"] < 0.2 * self.radii[-1]:
            total *= 0.2 * self.radii[-1] / self.par_values["r_break"]
        # Penalise huge bulge
        if self.bulge_to_total > 0.25:
            total *= 1+20*(self.bulge_to_total - 0.25)
        if self.par_values["re_b"] > 0.2 * self.radii[-1]:
            total *= 1+20*(self.par_values["re_b"] - 0.2*self.radii[-1])
        # Penalize small bulges Sersic values
        if self.par_values["n_b"] < 0.4:
            total *= 1 + 100*(0.4-self.par_values["n_b"])
        if self.old_bulge_re is not None:
            # Penalize bulge Re difference from the vertical fit
            diff = (self.par_values["re_b"] - self.old_bulge_re) / self.old_bulge_re
            if diff > 0.25:
                total *= 1 + 50*diff
        return total


class DoubleBrokenDiskFitter(BasicFitter):
    name = "Double broken expdisk"
    ref_points = ["Point on first disk", "First break point",
                  "Second break point", "Point on third disk"]
    n_free_params = 6

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        par_names = ("mu0_d1", "mu0_d2", "mu0_d3", "h_d1", "h_d2", "h_d3", "r_break1", "r_break2")
        par_dims = ("mag/sq.''", "''", "mag/sq.''", "''", "mag/sq.''", "''", "''", "''")
        par_free = (True, False, False, True, True, True, True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        mag = np.zeros_like(self.radii)
        inner = np.where(self.radii <= self.par_values["r_break1"])
        middle = np.where((self.par_values["r_break1"] < self.radii) * (self.radii <= self.par_values["r_break2"]))
        outer = np.where(self.radii > self.par_values["r_break2"])
        mag[inner] = self.par_values["mu0_d1"] + kappa * self.radii[inner] / self.par_values["h_d1"]
        mag[middle] = self.par_values["mu0_d2"] + kappa * self.radii[middle] / self.par_values["h_d2"]
        mag[outer] = self.par_values["mu0_d3"] + kappa * self.radii[outer] / self.par_values["h_d3"]
        m0 = 25
        flux_orig = 10 ** (0.4 * (m0 - mag))
        flux_orig_symmetric = np.append(flux_orig[::-1], flux_orig)  # Make symmetric for the padding purposes
        flux_convolved = fftconvolve(flux_orig_symmetric, self.psf, mode="same")[len(flux_orig):]
        return -2.5 * np.log10(flux_convolved) + m0

    def compute_initials(self, ref_points):
        h_d1 = kappa * (ref_points[0][0] - ref_points[1][0]) / (ref_points[0][1]-ref_points[1][1])
        h_d2 = kappa * (ref_points[1][0] - ref_points[2][0]) / (ref_points[1][1]-ref_points[2][1])
        h_d3 = kappa * (ref_points[2][0] - ref_points[3][0]) / (ref_points[2][1]-ref_points[3][1])
        mu0_d1 = ref_points[0][1] - kappa * ref_points[0][0] / h_d1
        r_break1 = ref_points[1][0]
        r_break2 = ref_points[2][0]
        self.set_params([mu0_d1, h_d1, h_d2, h_d3, r_break1, r_break2])

    def compute_non_free(self):
        self.par_values["mu0_d2"] = (self.par_values["mu0_d1"] +
                                     kappa * self.par_values["r_break1"] * (1/self.par_values["h_d1"]-
                                                                            1/self.par_values["h_d2"]))
        self.par_values["mu0_d3"] = (self.par_values["mu0_d2"] +
                                     kappa * self.par_values["r_break2"] * (1/self.par_values["h_d2"]-
                                                                            1/self.par_values["h_d3"]))

    def fit_auto(self):
        """
        Try to determine good initial parameters to and perform fit
        """
        mu0_d1 = self.surf_bri[0]
        r1 = len(self.radii) // 3
        r2 = 4 * len(self.radii) // 5
        step = (r2 - r1) // 15
        best_chi = 1e10
        s_interp = interp1d(self.radii, self.surf_bri, bounds_error=False)
        def f(breaks):
            r_break1 = breaks[0]
            r_break2 = breaks[1]
            h_d1 = kappa * (self.radii[0] - r_break1) / (self.surf_bri[0]-s_interp(r_break1))
            h_d2 = (kappa * (r_break1 - r_break2) /
                    (s_interp(r_break1)-s_interp(r_break2)))
            h_d3 = kappa * (r_break2 - self.radii[-1]) / (s_interp(r_break2)-self.surf_bri[-1])
            self.set_params([mu0_d1, h_d1, h_d2, h_d3, r_break1, r_break2])
            self.fit()
            return self.chi
        rmin = self.radii[0]
        rmax = self.radii[-1]
        x0 = (rmin + (rmax-rmin) / 3, rmin + 2*(rmax-rmin) / 3)
        best_params = fmin(f, x0=x0, disp=False)
        # self.set_params(*best_params)
        self.fit()

    def get_inner_disk_mu0(self):
        return self.par_values["mu0_d1"]

    def to_imfit(self, x_cen, y_cen, magzpt, image_scale):
        imfit = f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION DblBknExp3D\n"
        imfit += "PA  90  80,100\n"
        imfit += "inc  90  fixed\n"
        j0 = 10**(0.4 * (magzpt - self.par_values["mu0_d1"])) * image_scale**2 / (2*self.par_values["h_d1"])
        imfit += f"J_0 {j0}\n"
        h1 = self.par_values['h_d1']/image_scale
        if h1 < 0:
            h1 = 100
        h2 = self.par_values['h_d2']/image_scale
        if h2 < 0:
            h2 = 100
        h3 = self.par_values['h_d3']/image_scale
        if h3 < 0:
            h3 = 100
        imfit += f"h1 {h1}\n"
        imfit += f"h2 {h2}\n"
        imfit += f"h3 {h3}\n"
        imfit += f"r_break1 {self.par_values['r_break1']/image_scale}\n"
        imfit += f"r_break2 {self.par_values['r_break2']/image_scale}\n"
        imfit += "n 2  0.5,20\n"
        imfit += "z_0 $vertical_scale \n"
        return imfit

    def bic_penalty(self):
        total = 0
        # Penalise very short outer disk
        if (max(self.radii) - self.par_values["r_break2"]) / max(self.radii) < 0.2:
            total += 100
        elif len(np.where(self.radii > self.par_values["r_break2"])[0]) < 4:
            total += 100
        # Penalise if inner disk mimicing the bulge
        if self.par_values["r_break1"] < 0.2 * self.radii[-1]:
            total += 10
        # Penalise short inner disk
        if abs(self.par_values["r_break1"] - self.par_values["r_break2"]) / self.radii[-1] < 0.1:
            total += 20
        # Outer break out of bounds
        if self.par_values["r_break2"] > self.radii[-1]:
            total += 20
        # Penalise very short middle disk
        if (abs(self.par_values["r_break2"] - self.par_values["r_break1"])) / np.max(self.radii) < 0.1:
            total += 1000
        return total

    def chi_penalty(self):
        total = 1
        # Penalise if inner disk mimicing the bulge
        if self.par_values["r_break1"] < 0.2 * self.radii[-1]:
            total *= 0.2 * self.radii[-1] / self.par_values["r_break1"]
        # Penalize if r_break1 is bigger than r_break2
        if self.par_values["r_break1"] > self.par_values["r_break2"]:
            total *= 1 + 10*(self.par_values["r_break1"] - self.par_values["r_break2"]) / self.par_values["r_break1"]

        return total


class BulgeDoubleBrokenDiskFitter(BasicFitter):
    name = "Bulge + double broken disk"
    ref_points = ["First point on bulge", "Second point on bulge", "Third point on bulge",
                  "Point on first disk", "First break point", "Second break point",
                  "Point on third disk"]
    n_free_params = 9

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        par_names = ("re_b", "mu0_b", "n_b", "mu0_d1", "mu0_d2", "mu0_d3", "h_d1",
                     "h_d2", "h_d3", "r_break1", "r_break2")
        par_dims = ("''", "mag/sq.''", "", "mag/sq.''", "''", "mag/sq.''", "''",
                    "mag/sq.''", "''", "''", "''")
        par_free = (True, True, True, True, False, False, True, True, True, True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)
        self.old_bulge_re = None
        self.is_complex = True

    def fix_bulge_to(self, re_b, mu0_b, n_b):
        self.par_values["re_b"] = re_b
        self.par_values["mu0_b"] = mu0_b
        self.par_values["n_b"] = n_b
        self.par_free["re_b"] = False
        self.par_free["mu0_b"] = False
        self.par_free["n_b"] = False

    def free_bulge(self):
        self.old_bulge_re = self.par_values["re_b"]
        self.par_free["re_b"] = True
        self.par_free["mu0_b"] = True
        self.par_free["n_b"] = True

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        m0 = 25
        disk = np.zeros_like(self.radii)
        inner = np.where(self.radii <= self.par_values["r_break1"])
        middle = np.where((self.par_values["r_break1"] < self.radii) * (self.radii <= self.par_values["r_break2"]))
        outer = np.where(self.radii > self.par_values["r_break2"])
        disk[inner] = self.par_values["mu0_d1"] + kappa * self.radii[inner] / self.par_values["h_d1"]
        disk[middle] = self.par_values["mu0_d2"] + kappa * self.radii[middle] / self.par_values["h_d2"]
        disk[outer] = self.par_values["mu0_d3"] + kappa * self.radii[outer] / self.par_values["h_d3"]

        disk = 10**(0.4 * (m0 - disk))
        nu_b = 2 * self.par_values["n_b"] - 1/3 + 4 / (405*self.par_values["n_b"])
        bulge = 10**(0.4 * (m0 - (self.par_values["mu0_b"] +
                                  kappa * nu_b * (self.radii/self.par_values["re_b"]) ** (1/self.par_values["n_b"]))))
        week_bulge_idx = np.where(bulge < 0.01)[0]
        if len(week_bulge_idx):
            bulge[week_bulge_idx[0]:] = 0.01
        disk_flux = simps(disk*self.radii, self.radii)
        bulge_flux = simps(bulge*self.radii, self.radii)
        self.bulge_to_total = bulge_flux / (bulge_flux + disk_flux)
        total = disk + bulge
        if unpack is False:
            # Perform convolution of the total flux
            total_symmetric = np.append(total[::-1], total)
            total_convolved = fftconvolve(total_symmetric, self.psf, mode="same")[len(total):]
            return -2.5 * np.log10(total_convolved) + m0
        else:
            bulge_symmetric = np.append(bulge[::-1], bulge)
            bulge_convolved = fftconvolve(bulge_symmetric, self.psf, mode="same")[len(bulge):]
            disk_symmetric = np.append(disk[::-1], disk)
            disk_convolved = fftconvolve(disk_symmetric, self.psf, mode="same")[len(disk):]
            total_symmetric = np.append(total[::-1], total)
            total_convolved = fftconvolve(total_symmetric, self.psf, mode="same")[len(total):]
            bulge = -2.5 * np.log10(bulge_convolved) + m0
            disk = -2.5 * np.log10(disk_convolved) + m0
            total = -2.5 * np.log10(total_convolved) + m0
            return (bulge, disk, total)

    def compute_non_free(self):
        self.par_values["mu0_d2"] = (self.par_values["mu0_d1"] +
                                     kappa * self.par_values["r_break1"] * (1/self.par_values["h_d1"]-
                                                                            1/self.par_values["h_d2"]))
        self.par_values["mu0_d3"] = (self.par_values["mu0_d2"] +
                                     kappa * self.par_values["r_break2"] * (1/self.par_values["h_d2"]-
                                                                            1/self.par_values["h_d3"]))

    def compute_initials(self, ref_points):
        # Disk
        h_d1 = kappa * (ref_points[3][0] - ref_points[4][0]) / (ref_points[3][1]-ref_points[4][1])
        h_d2 = kappa * (ref_points[4][0] - ref_points[5][0]) / (ref_points[4][1]-ref_points[5][1])
        h_d3 = kappa * (ref_points[5][0] - ref_points[6][0]) / (ref_points[5][1]-ref_points[6][1])
        mu0_d1 = ref_points[3][1] - kappa * ref_points[3][0] / h_d1
        r_break1 = ref_points[4][0]
        r_break2 = ref_points[5][0]

        # Bulge
        m0 = 25
        f_total = 10 ** (0.4 * (m0 - ref_points[0][1]))
        f_disk = 10 ** (0.4 * (m0 - mu0_d1))
        f_bulge = f_total - f_disk
        if f_bulge > 0:
            mu0_b = -2.5 * np.log10(f_bulge) + m0
        else:
            mu0_b = mu0_d1
        re_b = ref_points[2][0] / 2
        k1 = (ref_points[1][1]-ref_points[0][1]) / (ref_points[1][0]-ref_points[0][0])
        k2 = (ref_points[2][1]-ref_points[1][1]) / (ref_points[2][0]-ref_points[1][0])
        if k1 < k2:
            n_b = 0.5
        else:
            n_b = 2

        self.set_params([re_b, mu0_b, n_b, mu0_d1, h_d1, h_d2, h_d3, r_break1, r_break2])

    def fit_auto(self):
        """
        Try to determine good initial parameters to and perform fit
        """
        # Suppose the bulge is located in the inner quarter of the slice. So lets
        # analyse outer three quarters for the double broken disk paraters and
        # then add the bulge to it
        disk_start_idx = len(self.radii) // 4
        mu0_d1 = self.surf_bri[disk_start_idx]
        s_interp = interp1d(self.radii, self.surf_bri, bounds_error=False)
        def f(breaks):
            r_break1 = breaks[0]
            r_break2 = breaks[1]
            h_d1 = kappa * (self.radii[disk_start_idx] - r_break1) / (self.surf_bri[0]-s_interp(r_break1))
            h_d2 = (kappa * (r_break1 - r_break2) /
                    (s_interp(r_break1)-s_interp(r_break2)))
            h_d3 = kappa * (r_break2 - self.radii[-1]) / (s_interp(r_break2)-self.surf_bri[-1])
            mu0_d2 = mu0_d1 + kappa * r_break1 * (1/h_d1 - 1/h_d2)
            mu0_d3 = mu0_d2 + kappa * r_break2 * (1/h_d2 - 1/h_d3)

            mag = np.zeros_like(self.radii)
            inner = np.where(self.radii <= r_break1)
            middle = np.where((r_break1 < self.radii) * (self.radii <= r_break2))
            outer = np.where(self.radii > r_break2)
            mag[inner] = mu0_d1 + kappa * self.radii[inner] / h_d1
            mag[middle] = mu0_d2 + kappa * self.radii[middle] / h_d2
            mag[outer] = mu0_d3 + kappa * self.radii[outer] / h_d3
            m0 = 25
            f_disk = 10 ** (0.4 * (m0 - mag))
            f_total = 10 ** (0.4 * (m0 - self.surf_bri))
            f_bulge = f_total - f_disk
            # Find bulge params
            if f_bulge[0] > 0:
                mu0_b_guess = -2.5 * np.log10(f_bulge[0]) + m0
                for i, f in enumerate(f_bulge):
                    if f < 0.666 * f_bulge[0]:
                        re_b_guess = self.radii[i]
                        break
                for i, f in enumerate(f_bulge):
                    if f < 0.333 * f_bulge[0]:
                        r_out = self.radii[i]
                if 2 * re_b_guess > r_out:
                    nb_guess = 0.5
                else:
                    nb_guess = 2.0
            else:
                re_b_guess = self.radii[1]
                mu0_b_guess = self.surf_bri[0] - 2
                nb_guess = 1.0
            self.set_params([re_b_guess, mu0_b_guess, nb_guess, mu0_d1, h_d1, h_d2, h_d3, r_break1, r_break2])
            self.fit()
            return self.chi

        rmin = self.radii[disk_start_idx]
        rmax = self.radii[-1]
        x0 = (rmin + (rmax-rmin) / 3, rmin + 2*(rmax-rmin) / 3)
        best_params = fmin(f, x0=x0, disp=False)
        self.fit()

    def get_inner_disk_mu0(self):
        return self.par_values["mu0_d1"]

    def to_imfit(self, x_cen, y_cen, magzpt, image_scale):
        nu_b = 2 * self.par_values["n_b"] - 1/3 + 4 / (405*self.par_values["n_b"])
        imfit = f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION Sersic\n"
        imfit += "PA  90 80,100\n"
        imfit += "ell  $bulge_ell   0.0,0.6\n"
        n_b = self.par_values['n_b']
        if n_b < 0.5:
            n_b = 0.5
        if n_b > 6:
            n_b = 6
        imfit += f"n {n_b}   0.3,8\n"
        mue = self.par_values["mu0_b"] + 2.5 * nu_b / np.log(10)
        ie = 10**(0.4*(magzpt - mue)) * image_scale**2
        imfit += f"I_e {ie}\n"
        imfit += f"r_e  {self.par_values['re_b']/image_scale}\n"
        imfit += f"X0  {x_cen}\n"
        imfit += f"Y0  {y_cen}\n"
        imfit += "FUNCTION DblBknExp3D\n"
        imfit += "PA  90  80,100\n"
        imfit += "inc  90  fixed\n"
        j0 = 10**(0.4 * (magzpt - self.par_values["mu0_d1"])) * image_scale**2 / (2*self.par_values["h_d1"])
        imfit += f"J_0 {j0}\n"
        h1 = self.par_values['h_d1']/image_scale
        if h1 < 0:
            h1 = 100
        h2 = self.par_values['h_d2']/image_scale
        if h2 < 0:
            h2 = 100
        h3 = self.par_values['h_d3']/image_scale
        if h3 < 0:
            h3 = 100
        imfit += f"h1 {h1}\n"
        imfit += f"h2 {h2}\n"
        imfit += f"h3 {h3}\n"
        imfit += f"r_break1 {self.par_values['r_break1']/image_scale}\n"
        imfit += f"r_break2 {self.par_values['r_break2']/image_scale}\n"
        imfit += "n 2  0.5,20\n"
        imfit += "z_0 $vertical_scale \n"
        return imfit

    def bic_penalty(self):
        # Penalise very short outer disk
        total = 0
        if (max(self.radii) - self.par_values["r_break2"]) / max(self.radii) < 0.1:
            total += 1000
        elif len(np.where(self.radii > self.par_values["r_break2"])[0]) < 4:
            total += 1000
        # Penalise very short middle disk
        if (abs(self.par_values["r_break2"] - self.par_values["r_break1"])) / np.max(self.radii) < 0.1:
            total += 1000
        return total

    def chi_penalty(self):
        # Penalise if inner disk is smaller than the bulge
        total = 1
        if self.par_values["r_break1"] < self.par_values["re_b"]:
            total *= abs(self.par_values["re_b"] / self.par_values["r_break1"])
        # Penalise if inner disk mimicing the bulge
        if self.par_values["r_break1"] < 0.2 * self.radii[-1]:
            total *= self.radii[-1] / self.par_values["r_break1"]
        # Penalize if r_break1 is bigger than r_break2
        if self.par_values["r_break1"] > self.par_values["r_break2"]:
            total *= 1 + 10*(self.par_values["r_break1"] - self.par_values["r_break2"]) / self.par_values["r_break1"]
        # Penalise huge bulge
        if self.bulge_to_total > 0.25:
            total *= 1+2*(self.bulge_to_total - 0.25)
        # Penalize small bulges Sersic values
        if self.par_values["n_b"] < 0.4:
            total *= 1 + 100*(0.4-self.par_values["n_b"])
        if self.old_bulge_re is not None:
            # Penalize bulge Re difference from the vertical fit
            diff = (self.par_values["re_b"] - self.old_bulge_re) / self.old_bulge_re
            if diff > 0.25:
                total *= 1 + 50*diff
        return total


class VerticalDiskFitter(BasicFitter):
    name = "Vertical disk"

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        # super().__init__(radii=radii, surf_bri=surf_bri, surf_bri_std=surf_bri_std, psf=psf)
        par_names = ("mu0_d", "z0_d", "c")
        par_dims = ("mag/sq.''", "''", "")
        par_free = (True, True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        m0 = 25
        f0 = 10 ** (0.4 * (m0 - self.par_values["mu0_d"]))
        flux_orig = f0 * (1/np.cosh(self.radii/self.par_values["z0_d"]))**2
        flux_orig_symmetric = np.append(flux_orig[::-1], flux_orig)  # Make symmetric for the padding purposes
        flux_convolved = fftconvolve(flux_orig_symmetric, self.psf, mode="same")[len(flux_orig):] + self.par_values["c"]
        return -2.5 * np.log10(flux_convolved) + m0

    def compute_initials(self, horizonthal_fitter):
        mu0_d = horizonthal_fitter.get_inner_disk_mu0()
        z = self.radii[-1]
        m0 = 25
        f_z = 10 ** (0.4 * (m0 - self.surf_bri[-1]))
        f_0 = 10 ** (0.4 * (m0 - mu0_d))
        z0_d = z / np.arccosh(1/(f_z/f_0)**0.5)
        self.set_params([mu0_d, z0_d, 0.0])


class BulgeVerticalDiskFitter(BasicFitter):
    name = "Bulge + vertical disk"

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        par_names = ("re_b", "mu0_b", "n_b", "mu0_d", "z0_d", "c")
        par_dims = ("''", "mag/sq.''", "", "mag/sq.''", "''", "")
        par_free = (True, True, True, True, True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)
        self.is_complex = True

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        m0 = 25
        nu_b = 2 * self.par_values["n_b"] - 1/3 + 4 / (405*self.par_values["n_b"])
        f0 = 10 ** (0.4 * (m0 - self.par_values["mu0_d"]))
        disk = f0 * (1/np.cosh(self.radii/self.par_values["z0_d"]))**2
        bulge_mag = (self.par_values["mu0_b"] +
                     kappa * nu_b * (self.radii/self.par_values["re_b"]) ** (1/self.par_values["n_b"]))
        bulge = 10 ** (0.4 * (m0 - bulge_mag))
        total = bulge + disk + self.par_values["c"]
        if unpack is False:
            # Perform convolution of the total flux
            total_symmetric = np.append(total[::-1], total)
            total_convolved = fftconvolve(total_symmetric, self.psf, mode="same")[len(total):]
            return -2.5 * np.log10(total_convolved) + m0
        else:
            bulge_symmetric = np.append(bulge[::-1], bulge)
            bulge_convolved = fftconvolve(bulge_symmetric, self.psf, mode="same")[len(bulge):]
            disk_symmetric = np.append(disk[::-1], disk)
            disk_convolved = fftconvolve(disk_symmetric, self.psf, mode="same")[len(disk):]
            total_symmetric = np.append(total[::-1], total)
            total_convolved = fftconvolve(total_symmetric, self.psf, mode="same")[len(total):]
            bulge = -2.5 * np.log10(bulge_convolved) + m0
            disk = -2.5 * np.log10(disk_convolved) + m0
            total = -2.5 * np.log10(total_convolved) + m0
            return (bulge, disk, total)

    def compute_initials(self, horizonthal_fitter):
        # Disk
        mu0_d = horizonthal_fitter.get_inner_disk_mu0()
        z = self.radii[-1]
        m0 = 25
        f_z = 10 ** (0.4 * (m0 - self.surf_bri[-1]))
        f_0 = 10 ** (0.4 * (m0 - mu0_d))
        z0_d = z / np.arccosh(1/(f_z/f_0)**0.5)
        # Bulge
        mu0_b = horizonthal_fitter.par_values["mu0_b"]
        re_b = horizonthal_fitter.par_values["re_b"]
        n_b = horizonthal_fitter.par_values["n_b"]
        self.set_params([re_b, mu0_b, n_b, mu0_d, z0_d, 0.0])


class AddVerticalDiskFitter(BasicFitter):
    name = "Vertical disk"

    def __init__(self, radii, surf_bri, surf_bri_std, psf):
        # super().__init__(radii=radii, surf_bri=surf_bri, surf_bri_std=surf_bri_std, psf=psf)
        par_names = ("r_c", "mu0_d", "z0_d", "c")
        par_dims = ("''", "mag/sq.''", "''", "")
        par_free = (True, True, True, True)
        super().__init__(radii, surf_bri, surf_bri_std, psf, par_names, par_dims, par_free)

    def evaluate(self, unpack=False):
        if None in self.get_par_values():
            return None
        m0 = 25
        f0 = 10 ** (0.4 * (m0 - self.par_values["mu0_d"]))
        flux_orig = f0 * (1/np.cosh((self.radii-self.par_values["r_c"])/self.par_values["z0_d"]))**2
        flux_convolved = fftconvolve(flux_orig, self.psf, mode="same") + self.par_values["c"]
        return -2.5 * np.log10(flux_convolved) + m0

    def compute_initials(self, horizonthal_fitter):
        mu0_d = self.surf_bri[len(self.surf_bri)//2]
        z = self.radii[-1]
        m0 = 25
        f_z = 10 ** (0.4 * (m0 - self.surf_bri[-1]))
        f_0 = 10 ** (0.4 * (m0 - mu0_d))
        z0_d = 0.75 * z / np.arccosh(1/(f_z/f_0)**0.5)
        r_c = self.radii[len(self.radii) // 2]
        self.set_params([r_c, mu0_d, z0_d, 0.0])
