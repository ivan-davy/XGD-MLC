from config import settings


class Peak:
    def __init__(self, isotope, line_kev):
        self.isotope = isotope
        self.line_kev = line_kev
        self.left_b = self.line_kev - settings.peak_delta_x
        self.right_b = self.line_kev + settings.peak_delta_x


