from datetime import date
from classes.isoclass import Isotope

clf_proba_custom_multipliers = {  # >: more detections, <: less detections
    'Na22': 10,
    'Co60': 5,
    'Cs137': 5,
    'Eu152': 2,
    'Am241': 15
}

clf_isotopes = {
    'Na22': Isotope(0, 'Sodium-22', 22, 11, 2.602, 107000, date(2016, 10, 1), date(2022, 7, 1),
                    [511, 1275], '#1803ff'),
    'Co60': Isotope(1, 'Cobalt-60', 60, 27, 5.271, 100000, date(2016, 10, 1), date(2022, 7, 1),
                    [1173, 1333], '#fab732'),
    'Cs137': Isotope(2, 'Caesium-137', 137, 55, 30.05, 116000, date(2016, 10, 1), date(2022, 7, 1),
                     [662], '#ff0303'),
    'Eu152': Isotope(3, 'Europium-152', 152, 63, 13.516, 104000, date(2016, 10, 1), date(2022, 10, 1),
                     [122, 245, 344, 411, 444, 779, 867, 964, 1090, 1408], '#ea18ed'),  # +1086
    'Am241': Isotope(4, 'Americium-241', 241, 95, 432.2, 111000, date(2016, 10, 1), date(2022, 7, 1),
                     [60], '#03f7ff'),
}

test_sps_acquisition_date = date(2022, 7, 1)

cal_area_to_act_multiplier_distance = 15  # R(cm)
cal_area_to_act_multiplier = {  # {Isotope: A(Bq) / S = k = multiplier}
    'Na22': 1050,
    'Co60': 6900,
    'Cs137': 3600,
    'Eu152': 390,
    'Am241': 14350
}
cal_act_uncertainty_threshold_kBq = 200  # Considers activities higher than this unrealistic (strictly visual)
