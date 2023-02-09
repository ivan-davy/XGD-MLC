from scipy.stats import stats
from classes.peakclass import Peak
import config.isodata
from mlclf.binary import *
from config import settings
import numpy as np


class Spectrum:
    def __init__(self, path, channel_qty, real_time_int, live_time_int, cal, bin_data):
        self.path = path
        self.channel_qty = channel_qty
        self.real_time_int = real_time_int
        self.live_time_int = live_time_int
        self.cal = cal
        self.bin_data = bin_data
        self.calib_bins = None
        self.calib_bin_data = None
        self.rebin_bins = None
        self.rebin_bin_data = None
        self.count_rate_bin_data = None
        self.features_array = None
        self.has_source = None
        self.src_known_isotope = None
        self.peak_data = None
        self.corrupted = False

    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            ('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def calibrate_naive(self):
        if not self.corrupted:
            x = [i * self.cal[0] + self.cal[1] for i in range(self.channel_qty)]
            y = [self.bin_data[i] / self.cal[0] for i in range(len(self.bin_data))]
            if not settings.keep_redundant_data:
                x = x[:int(settings.kev_cap * self.cal[0])]
                y = y[:len(x)]
            self.calib_bins = x
            self.calib_bin_data = y

    def calibrate(self):
        if not self.corrupted:
            x = (np.arange(self.channel_qty) * self.cal[0] + self.cal[1]).tolist()
            y = (np.array(self.bin_data) / self.cal[0]).tolist()
            if not settings.keep_redundant_data:
                x = x[:int(settings.kev_cap * self.cal[0])]
                y = y[:len(x)]
            self.calib_bins = x
            self.calib_bin_data = y
        return self

    def rebin_naive(self):
        self.calibrate()
        if not self.corrupted:
            x_old = self.calib_bins
            max_kev = int(x_old[-1])
            x_new = np.arange(max_kev).tolist()  # type
            self.rebin_bins = x_new
            bar_events_old = self.calib_bin_data
            grid, slices_in_old, slices_in_new = [], [], []
            iter_new, iter_old = 0, 0
            while iter_new <= max_kev - 1 and iter_old != self.channel_qty - 1:
                if x_old[iter_old] <= x_new[iter_new]:
                    if x_old[iter_old] not in grid:
                        grid.append(x_old[iter_old])
                    slices_in_old.append(iter_old)
                    if x_old[iter_old] < self.cal[1]:
                        slices_in_new.append(None)
                    else:
                        slices_in_new.append(iter_new - 1)
                    iter_old += 1
                elif x_new[iter_new] < x_old[iter_old]:
                    if x_new[iter_new] not in grid:
                        grid.append(x_new[iter_new])
                    slices_in_new.append(iter_new)
                    if x_new[iter_new] < self.cal[1]:
                        slices_in_old.append(None)
                    else:
                        slices_in_old.append(iter_old - 1)
                    iter_new += 1
            slices = [] * len(grid)
            for i in range(len(grid) - 1):
                if slices_in_old[i] is not None:
                    slices.append(bar_events_old[slices_in_old[i]] * (grid[i + 1] - grid[i]))
                else:
                    slices.append(int(0))
            rebin_bin_data = [0] * max_kev
            for i in range(max_kev - 1):
                if slices_in_new[i] is not None:
                    rebin_bin_data[slices_in_new[i]] += slices[i]
            del rebin_bin_data[max_kev:]
            if settings.keep_redundant_data is False:
                rebin_bin_data = rebin_bin_data[:int(settings.kev_cap)]
            self.rebin_bin_data = rebin_bin_data
        return self

    def rebin(self):  # adapted version of https://github.com/jhykes/rebin
        self.calibrate()
        if not self.corrupted:
            self.rebin_bins = np.arange(self.calib_bins[-1]).tolist()
            x1 = np.asarray(self.calib_bins)
            y1 = np.asarray(self.calib_bin_data)
            x2 = np.asarray(np.arange(self.calib_bins[-1]))

            # the fractional bin locations of the new bins in the old bins
            i_place = np.interp(x2, x1, np.arange(len(x1)))
            cum_sum = np.r_[[0], np.cumsum(y1)]

            # calculate bins where lower and upper bin edges span
            # greater than or equal to one original bin.
            # This is the contribution from the 'intact' bins (not including the
            # fractional start and end parts).
            whole_bins = np.floor(i_place[1:]) - np.ceil(i_place[:-1]) >= 1.
            start = cum_sum[np.ceil(i_place[:-1]).astype(int)]
            finish = cum_sum[np.floor(i_place[1:]).astype(int)]

            y2 = np.where(whole_bins, finish - start, 0.)

            bin_loc = np.clip(np.floor(i_place).astype(int), 0, len(y1) - 1)

            # fractional contribution for bins whose new bin edges are in the same
            # original bin.
            same_cell = np.floor(i_place[1:]) == np.floor(i_place[:-1])
            frac = i_place[1:] - i_place[:-1]
            contrib = (frac * y1[bin_loc[:-1]])
            y2 += np.where(same_cell, contrib, 0.)

            # fractional contribution for bins whose left and right bin edges are in
            # different original bins.
            different_cell = np.floor(i_place[1:]) > np.floor(i_place[:-1])
            frac_left = np.ceil(i_place[:-1]) - i_place[:-1]
            contrib = (frac_left * y1[bin_loc[:-1]])

            frac_right = i_place[1:] - np.floor(i_place[1:])
            contrib += (frac_right * y1[bin_loc[1:]])

            y2 += np.where(different_cell, contrib, 0.)
            self.rebin_bin_data = (y2 * self.cal[0]).tolist()
        return self

    def calcCountRate(self):
        if not self.corrupted:
            count_rate_bin_data = [self.rebin_bin_data[i] / self.live_time_int for i in range(len(self.rebin_bin_data))]
            setattr(self, 'count_rate_bin_data', count_rate_bin_data)
        self.cleanup()
        return self

    def cleanup(self):
        if self.corrupted:
            del self
        elif settings.keep_redundant_data is False:
            del self.calib_bins
            del self.calib_bin_data
            if settings.kev_cap != 0:
                del self.rebin_bins[settings.kev_cap:]
                del self.rebin_bin_data[settings.kev_cap:]
                del self.count_rate_bin_data[settings.kev_cap:]

    def subtractCountRateBkg(self, bkg):
        if bkg.count_rate_bin_data is None:
            bkg.calcCountRate()
        self.count_rate_bin_data = [self.count_rate_bin_data[i] - bkg.count_rate_bin_data[i]
                                    if self.count_rate_bin_data[i] - bkg.count_rate_bin_data[i] >= 0 else 0.0
                                    for i in range(len(self.count_rate_bin_data))]

    def getNumOfEvents(self, dtype='count_rate'):
        if dtype == 'raw':
            return sum(self.bin_data)
        if dtype == 'rebin':
            return sum(self.bin_data)
        if dtype == 'count_rate':
            return sum(self.count_rate_bin_data)

    def generatePeaksData(self, known_isotopes_names):
        self.peak_data = {}
        for isotope_name in known_isotopes_names:
            peaks = {}
            lines = config.isodata.clf_isotopes[isotope_name].lines
            for line_kev in lines:
                peaks[line_kev] = Peak(self, isotope_name, line_kev)
            self.peak_data[isotope_name] = peaks

    def sigmaBinaryClassify(self, bkg, thr_multiplier=3):  # sigma
        self.rebin().calcCountRate()
        sp_avg = sum(self.count_rate_bin_data)
        sp_err = (sp_avg * self.live_time_int) ** 0.5 / self.live_time_int
        bkg_avg = sum(bkg.count_rate_bin_data)
        bkg_err = (bkg_avg * bkg.live_time_int) ** 0.5 / bkg.live_time_int
        diff_avg = abs(sp_avg - bkg_avg)
        diff_err = (sp_err ** 2 + bkg_err ** 2) ** 0.5

        if diff_avg > diff_err * thr_multiplier:
            return 'Source'
        else:
            return 'Background'

    def pearsonBinaryClassify(self, bkg, num_of_sections=settings.bin_clf_sections_qty,
                              tolerance=1, los=0.05):  # Ineffective method, not implemented
        bkg_counts_per_section = sum(bkg.count_rate_bin_data) / num_of_sections
        bkg_sections_avg, bkg_sections_err, section_borders, bin_iter, sec_iter, temp_sum = [], [], [0, ], 0, 0, 0
        for section in range(num_of_sections):
            while temp_sum < bkg_counts_per_section and bin_iter != len(bkg.rebin_bins) - 1:
                temp_sum += bkg.count_rate_bin_data[bin_iter]
                bin_iter += 1
            bkg_sections_avg.append(temp_sum)
            section_borders.append(bin_iter)
            temp_sum = 0
        sp_sections_avg, sp_sections_err, bin_iter, sec_iter, temp_sum = [], [], 0, 0, 0
        for section in range(num_of_sections):
            while bin_iter < section_borders[section + 1] and bin_iter != len(self.rebin_bins) - 1:
                temp_sum += self.count_rate_bin_data[bin_iter]
                bin_iter += 1
            sp_sections_avg.append(temp_sum)
            temp_sum = 0
        chi = 0
        for section in range(num_of_sections):
            chi += (sp_sections_avg[section] - bkg_sections_avg[section]) ** 2 / bkg_sections_avg[section]
        if chi > stats.chi2.ppf(1 - los, num_of_sections) * tolerance:  # 0.05 => 0.95
            return 'Source'
        else:
            return 'Background'

    def chi2BinaryClassify(self, bkg, num_of_sections=settings.bin_clf_sections_qty,
                           tolerance=3, los=0.05):  # Ineffective method, not implemented
        bkg_counts_per_section = sum(bkg.count_rate_bin_data) / num_of_sections
        bkg_sections_avg, bkg_sections_err, section_borders, bin_iter, sec_iter, temp_sum = [], [], [0, ], 0, 0, 0
        for section in range(num_of_sections):
            while temp_sum < bkg_counts_per_section and bin_iter != len(bkg.rebin_bins) - 1:
                temp_sum += bkg.count_rate_bin_data[bin_iter]
                bin_iter += 1
            bkg_sections_avg.append(temp_sum)
            bkg_sections_err.append((temp_sum * bkg.live_time_int) ** 0.5 / bkg.live_time_int)
            section_borders.append(bin_iter)
            temp_sum = 0
        sp_sections_avg, sp_sections_err, bin_iter, sec_iter, temp_sum = [], [], 0, 0, 0
        for section in range(num_of_sections):
            while bin_iter < section_borders[section + 1] and bin_iter != len(self.rebin_bins) - 1:
                temp_sum += self.count_rate_bin_data[bin_iter]
                bin_iter += 1
            sp_sections_avg.append(temp_sum)
            sp_sections_err.append((temp_sum * self.live_time_int) ** 0.5 / self.live_time_int)
            temp_sum = 0

        chi = 0
        for section in range(num_of_sections):
            chi += (sp_sections_avg[section] - bkg_sections_avg[section]) ** 2 \
                   / (sp_sections_err[section] ** 2 + bkg_sections_err[section] ** 2)
        if chi > stats.chi2.ppf(1 - los, num_of_sections) * tolerance:  # 0.05 => 0.95
            return 'Source'
        else:
            return 'Background'
