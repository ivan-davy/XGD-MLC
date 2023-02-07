from config import settings, isodata
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

    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            ('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


