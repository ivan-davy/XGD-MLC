import numpy as np
from scipy import stats
from config import *
from mlbinclf import *


class Spectrum:
    def __init__(self, location, channel_qty, real_time_int, live_time_int, cal, bin_data):
        self.location = location
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
        self.isotope = None
        self.corrupted = False

    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            ('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def calibrate(self):
        if not self.corrupted:
            x = [i * self.cal[0] + self.cal[1] for i in range(self.channel_qty)]
            y = [self.bin_data[i] / self.cal[0] for i in range(len(self.bin_data))]
            if keep_redundant_data is False:
                x = x[:int(kev_cap * self.cal[0])]
                y = y[:len(x)]
            self.calib_bins = x
            self.calib_bin_data = y

    def rebin(self):
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
            if keep_redundant_data is False:
                rebin_bin_data = rebin_bin_data[:int(kev_cap)]
            self.rebin_bin_data = rebin_bin_data
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
        elif keep_redundant_data is False:
            del self.calib_bins
            del self.calib_bin_data
            if kev_cap != 0:
                del self.rebin_bins[kev_cap:]
                del self.rebin_bin_data[kev_cap:]
                del self.count_rate_bin_data[kev_cap:]

    def subtractBkg(self, bkg):
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

    def pearsonBinaryClassify(self, bkg, num_of_sections=bin_clf_sections_qty,  # Ineffective method, not implemented
                              tolerance=1, los=0.05):
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

    def chi2squareBinaryClassify(self, bkg, num_of_sections=bin_clf_sections_qty,  # Ineffective method, not implemented
                                 tolerance=3, los=0.05):
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
