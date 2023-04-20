from config import settings
from config.isodata import clf_isotopes
from simple_chalk import chalk


def checkTotalEvents(spectrum):
    print(chalk.cyan('Total area: '))
    s = 0
    for i in range(spectrum.channel_qty):
        s += spectrum.bin_data[i]
    print(chalk.cyan('Uncalibrated: ', s))
    if settings.keep_redundant_data is True:
        s = 0
        for i in range(len(spectrum.calib_bins) - 1):
            s += (spectrum.calib_bins[i + 1] - spectrum.calib_bins[i]) * spectrum.calib_bin_data[i]
        print(chalk.cyan('Calibrated: ', s))
    s = 0
    for i in range(len(spectrum.rebin_bins) - 1):
        s += (spectrum.rebin_bins[i + 1] - spectrum.rebin_bins[i]) * spectrum.rebin_bin_data[i]
    print(chalk.cyan('Rebinned: ', s))


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
    print(chalk.cyan(f'\nAccuracy ({method}, {tolerance=}, {los=}):'), f'{accuracy}')
    print([tp, fp], '\n', [fn, tn])
    return [[tp, fp], [fn, tn]], accuracy


def getClfMetrics(results, **user_args):
    print(chalk.blue('\nCalculating statistics...'),
          f'(clf_threshold = {chalk.cyan(settings.clf_threshold)})')

    #  results = {key: val for key, val in results.items() if '60s' in key}

    matches = 0
    for key, val in results.items():
        match = True
        for isotope in clf_isotopes.keys():
            if isotope in key:
                if not (val.get(isotope, 0) > settings.clf_threshold):
                    match = False
                    break
        if match:
            matches += 1

    emr_result = matches / len(results.keys())
    print(chalk.cyan(f'EMR:'), emr_result)

    total_sum_precision = 0
    total_sum_recall = 0
    for key, val in results.items():
        correctly_guessed = 0
        guessed = 0
        expected_isotopes = 0
        precision = 0
        recall = 0
        for isotope in clf_isotopes.keys():
            if val.get(isotope, 0) > settings.clf_threshold:
                guessed += 1
            if isotope in key:
                expected_isotopes += 1
                if val.get(isotope, 0) > settings.clf_threshold:
                    correctly_guessed += 1
        if expected_isotopes != 0:
            precision = correctly_guessed / expected_isotopes
        if guessed != 0:
            recall = correctly_guessed / guessed
        total_sum_precision += precision
        total_sum_recall += recall

    precision_result = total_sum_precision / len(results.keys())
    print(chalk.cyan(f'Precision:'), precision_result)
    recall_result = total_sum_recall / len(results.keys())
    print(chalk.cyan(f'Recall:'), recall_result)

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
            if val.get(isotope, 0) > settings.clf_threshold:
                unique_isotopes.add(isotope)
        for unique_isotope in unique_isotopes:
            per_isotope_total[unique_isotope] += 1
            if val.get(unique_isotope, 0) > settings.clf_threshold and unique_isotope in key:
                correctly_guessed += 1
                if unique_isotope in key:
                    per_isotope_correctly_guessed[unique_isotope] += 1
        accuracy = correctly_guessed / len(unique_isotopes)
        total_sum += accuracy

    accuracy_result = total_sum / len(results.keys())
    print(chalk.cyan(f'Accuracy:'), accuracy_result)

    f1_result = 2 * precision_result * recall_result / (precision_result + recall_result)
    print(chalk.cyanBright(f'F1-Score:'), f1_result)

    print(chalk.cyan('\nPer-isotope accuracies:'))
    for key in per_isotope_total:
        try:
            per_isotope_accuracies[key] = per_isotope_correctly_guessed[key] / per_isotope_total[key]
            print(f'{key}: {per_isotope_accuracies[key]}')
        except ZeroDivisionError:
            continue

    return emr_result, precision_result, recall_result, accuracy_result, f1_result
