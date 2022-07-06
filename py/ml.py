import config
from visual import mlShowLinfit, mlShowAverage
from utility import loadSpectrumData


def mlLoadSets():
    from os import walk, path
    source_ml_set, background_ml_set = [], []
    for root, dirs, files in walk(config.src_fileset_location):
        for file in files:
            if file.endswith('.sps'):
                source_ml_set.append(loadSpectrumData(path.join(root, file)))
    for root, dirs, files in walk(config.bkg_fileset_location):
        for file in files:
            if file.endswith('.sps'):
                background_ml_set.append(loadSpectrumData(path.join(root, file)))
    return source_ml_set, background_ml_set


def mlGetFeatures(ml_set, feature_type, bins_per_sect=config.ml_bin_clf_bins_per_section,
                  show_progress=True,
                  show=False):
    num_of_sections = int(config.kev_cap / bins_per_sect)
    if feature_type == 'linfit':
        from visual import linear
        from scipy.optimize import curve_fit
        set_a, set_b, counter = [], [], 0
        for spectrum in ml_set:
            counter += 1
            spectrum_a, spectrum_b = [], []
            if spectrum.corrupted is False:
                spectrum.rebin()
                spectrum.calcCountRate()
            else:
                continue
            for section in range(num_of_sections):
                section_ab = curve_fit(linear,
                                       spectrum.rebin_bins[section * bins_per_sect:(section + 1) * bins_per_sect],
                                       spectrum.count_bin_data[section * bins_per_sect:(section + 1) * bins_per_sect])
                spectrum_a.append(section_ab[0][0])
                spectrum_b.append(section_ab[0][1])
            if show:
                mlShowLinfit(spectrum, spectrum_a, spectrum_b, bins_per_sect)
            set_a.append(spectrum_a)
            set_b.append(spectrum_b)
            if show_progress:
                print('\r', counter, '/', len(ml_set), spectrum.location, end='')
        if show_progress:
            print('\n')
        return set_a, set_b

    elif feature_type == 'average':
        set_c, counter = [], 0
        for spectrum in ml_set:
            counter += 1
            spectrum_c = []
            if spectrum.corrupted is False:
                spectrum.rebin()
                spectrum.calcCountRate()
            else:
                continue
            for section in range(num_of_sections):
                section_c = spectrum.count_bin_data[section * bins_per_sect:(section + 1) * bins_per_sect]
                spectrum_c.append(sum(section_c) / bins_per_sect)
            if show:
                mlShowAverage(spectrum, spectrum_c, bins_per_sect)
            set_c.append(spectrum_c)
            if show_progress:
                print('\r', counter, '/', len(ml_set), spectrum.location, end='')
        if show_progress:
            print('\n')
        return set_c
