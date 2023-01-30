from config import settings
from config.isodata import clf_isotopes


def checkTotalEvents(spectrum):
    print('Total area:')
    s = 0
    for i in range(spectrum.channel_qty):
        s += spectrum.bin_data[i]
    print('Uncalibrated: ', s)
    if settings.keep_redundant_data is True:
        s = 0
        for i in range(len(spectrum.calib_bins) - 1):
            s += (spectrum.calib_bins[i + 1] - spectrum.calib_bins[i]) * spectrum.calib_bin_data[i]
        print('Calibrated: ', s)
    s = 0
    for i in range(len(spectrum.rebin_bins) - 1):
        s += (spectrum.rebin_bins[i + 1] - spectrum.rebin_bins[i]) * spectrum.rebin_bin_data[i]
    print('Rebinned: ', s)


def getBinaryConfusionMatrix(srcs, bkgs, bkg_sp_reference, method, tolerance, los):
    res, tp, fp, fn, tn = None, 0, 0, 0, 0
    for spectrum in srcs:
        if not spectrum.corrupted:
            if method == 'sigma':
                res = spectrum.sigmaBinaryClassify(bkg_sp_reference)
            elif method == 'pearson':
                res = spectrum.pearsonBinaryClassify(bkg_sp_reference, tolerance=tolerance, los=los)
            elif method == 'chi2square':
                res = spectrum.chi2BinaryClassify(bkg_sp_reference, tolerance=tolerance, los=los)
            if res == 'Signal':
                tp += 1
            if res == 'Background':
                fp += 1
    print('')
    for spectrum in bkgs:
        if not spectrum.corrupted:
            if method == 'sigma':
                res = spectrum.sigmaBinaryClassify(bkg_sp_reference)
            elif method == 'pearson':
                res = spectrum.pearsonBinaryClassify(bkg_sp_reference, tolerance=tolerance, los=los)
            elif method == 'chi2square':
                res = spectrum.chi2BinaryClassify(bkg_sp_reference, tolerance=tolerance, los=los)
            if res == 'Signal':
                fn += 1
            if res == 'Background':
                tn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f'\nAccuracy ({method}, {tolerance=}, {los=}): {accuracy}')
    print([tp, fp], '\n', [fn, tn])
    return [[tp, fp], [fn, tn]], accuracy


def getClfMetrics(results, show_results=True, **user_args):
    print(f'Calculating statistics... (clf_threshold={settings.clf_threshold})\n')

    matches = 0
    for key, val in results.items():
        match = True
        for isotope in clf_isotopes.keys():
            if isotope in key:
                if not (val[isotope] > settings.clf_threshold):
                    match = False
                    break
        if match:
            matches += 1

    total_sum = 0
    for key, val in results.items():
        correctly_guessed = 0
        expected_isotopes = 0
        precision = 0
        for isotope in clf_isotopes.keys():
            if isotope in key:
                expected_isotopes += 1
                if val[isotope] > settings.clf_threshold:
                    correctly_guessed += 1
        if expected_isotopes != 0:
            precision = correctly_guessed/expected_isotopes
        total_sum += precision

    total_sum = 0
    per_isotope_correctly_guessed = dict.fromkeys(clf_isotopes, 0)
    per_isotope_total = dict.fromkeys(clf_isotopes, 0)
    per_isotope_accuracies = {}
    for key, val in results.items():
        correctly_guessed = 0
        unique_isotopes = set()
        for isotope in clf_isotopes.keys():
            if isotope in key:
                unique_isotopes.add(isotope)
            if val[isotope] > settings.clf_threshold:
                unique_isotopes.add(isotope)
        for unique_isotope in unique_isotopes:
            per_isotope_total[unique_isotope] += 1
            if val[unique_isotope] > settings.clf_threshold:
                correctly_guessed += 1
                per_isotope_correctly_guessed[unique_isotope] += 1
        accuracy = correctly_guessed/len(unique_isotopes)
        total_sum += accuracy

    if show_results:
        print(f'EMR: {matches/len(results.keys())}')
        print(f'Precision: {total_sum/len(results.keys())}')
        print(f'Accuracy: {total_sum/len(results.keys())}')

        print('\nPer-isotope accuracies:')
        for key in per_isotope_total:
            per_isotope_accuracies[key] = per_isotope_correctly_guessed[key] / per_isotope_total[key]
            print(f'{key}: {per_isotope_accuracies[key]}')