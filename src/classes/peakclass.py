from config import settings, isodata
from utility import common
import numpy as np


class Peak:
    def __init__(self, spectrum, isotope_name, line_kev):
        self.spectrum = spectrum
        self.isotope = isodata.clf_isotopes[isotope_name]
        self.line_kev = line_kev
        self.left_b = self.line_kev - settings.peak_delta_x
        self.right_b = self.line_kev + settings.peak_delta_x
        self.baseline_linear_params = np.polyfit([self.left_b, self.right_b],
                                                 [spectrum.count_rate_bin_data[self.left_b],
                                                  spectrum.count_rate_bin_data[self.right_b]], 1)
        self.raw_area = sum(spectrum.count_rate_bin_data[self.left_b:self.right_b])
        self.area = sum([spectrum.count_rate_bin_data[kev] - self.calc_bin_under_baseline(kev)
                         for kev in range(self.left_b, self.right_b)])

    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            ('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def calc_bin_under_baseline(self, x):
        spectrum_y = self.spectrum.count_rate_bin_data[x]
        line_y = common.linear(x, self.baseline_linear_params[0], self.baseline_linear_params[1])
        return line_y if line_y <= spectrum_y else spectrum_y
