from datetime import date
from isoclass import Isotope


clf_proba_custom_multipliers = {  # >: more detections, <: less detections
    'Na22': 2,
    'Co60': 1,
    'Cs137': 1,
    'Am241': 10
}

clf_isotopes = {
    'Na22': Isotope(0, 'Sodium-22 ', 22, 11, 2.602, 107000, date(2016, 10, 1), date(2022, 7, 1),
                    [511, 1274.5], '#1803ff'),
    'Co60': Isotope(1, 'Cobalt-60 ', 60, 27, 5.271, 100000, date(2016, 10, 1), date(2022, 7, 1),
                    [1173.2, 1332.5], '#d98518'),
    'Cs137': Isotope(2, 'Caesium-137 ', 137, 55, 30.05, 116000, date(2016, 10, 1), date(2022, 7, 1),
                     [32, 662], '#ff0303'),
    'Am241': Isotope(3, 'Americium-241 ', 241, 95, 432.2, 111000, date(2016, 10, 1), date(2022, 7, 1),
                     [59.6], '#03f7ff')
}

test_sps_acquisition_date = date(2022, 7, 1)
