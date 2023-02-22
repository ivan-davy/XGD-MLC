from config import isodata, settings


def predictActivity(spectrum, confirmed_isotopes):
    spectrum.generatePeaksData(confirmed_isotopes)
    isotope_peak_sums = {}
    for isotope, val in spectrum.peak_data.items():
        area = 0
        for kev, peak in val.items():
            area += peak.area
        isotope_peak_sums[isotope] = area

    predicted_activities = {}
    spectrum.distance_to_src = settings.default_distance_to_src if not spectrum.distance_to_src \
        else spectrum.distance_to_src
    for key, val in isotope_peak_sums.items():
        predicted_activities[key] = val * isodata.cal_area_to_act_multiplier[key] \
                                    * ((spectrum.distance_to_src + settings.detector_inner_r) /
                                       (isodata.cal_area_to_act_multiplier_distance + settings.detector_inner_r)) \
                                    ** 2
    return predicted_activities
