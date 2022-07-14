from datetime import date


class Isotope:
    def __init__(self, iso_id, name, A, Z, half_life_yrs, original_decay, original_date, acquisition_date):
        self.iso_id = iso_id
        self.name = name
        self.A = A
        self.Z = Z
        self.half_life = half_life_yrs
        self.original_decay = original_decay
        self.original_date = original_date
        self.acquisition_date = acquisition_date
        self.acquisition_decay = self.original_decay * 2 ** (-(self.acquisition_date - self.original_date).days
                                                             / (365.25 * self.half_life))
