import config
from datetime import date


class Isotope:
    def __init__(self, name, A, Z, half_life_yrs, orig_decay, orig_year, orig_month, orig_day):
        self.name = name
        self.A = A
        self.Z = Z
        self.half_life = half_life_yrs
        self.original_decay = orig_decay
        self.original_date = date(orig_year, orig_month, orig_day)
        self.current_decay = orig_decay * 2 ** (-(config.sps_acquisition_date - self.original_date) / self.half_life)
