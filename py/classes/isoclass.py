class Isotope:
    def __init__(self, iso_id, name, mass_number, atomic_number,
                 half_life_yrs, original_decay_rate,
                 original_date, acquisition_date,
                 lines, color):
        self.iso_id = iso_id
        self.name = name
        self.A = mass_number
        self.Z = atomic_number
        self.half_life = half_life_yrs
        self.orig_decay_rate = original_decay_rate
        self.orig_date = original_date
        self.acq_date = acquisition_date
        self.acq_decay_rate = original_decay_rate * 2 ** (-(acquisition_date - original_date).days
                                                          / (365.25 * self.half_life))
        self.lines = lines
        self.color = color
