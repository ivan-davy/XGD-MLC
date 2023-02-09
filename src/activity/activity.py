def predictActivity(spectrum, confirmed_isotopes):
    spectrum.generatePeaksData(confirmed_isotopes)
    isotope_peak_sums = {}
    for isotope, val in spectrum.peak_data.items():
        area = 0
        for kev, peak in val.items():
            area += peak.area
        isotope_peak_sums[isotope] = area


