from struct import unpack_from
import config
from spclass import Spectrum


def loadSpectrumData(sps_filename):
    with open(sps_filename, 'rb') as file:
        buffer = file.read()
        live_time_int = (unpack_from('i', buffer, 301))[0]
        real_time_int = (unpack_from('i', buffer, 305))[0]
        num_of_bins = (unpack_from('h', buffer, 0))[0]
        calibration = unpack_from('ff', buffer, 440)
        bin_data = unpack_from(str(num_of_bins) + 'i', buffer, 1024)
    spectrum = Spectrum(sps_filename, num_of_bins, real_time_int, live_time_int, calibration, bin_data)
    if not spectrum.bin_data:
        spectrum.corrupted = True
        print(spectrum.location)
    if spectrum.cal[0] == 0:
        if config.enforce_cal:
            spectrum.cal = config.default_cal
        else:
            spectrum.corrupted = True
    return spectrum


def exportSpectrumData(filename, x_data, y_data):
    with open(filename, 'w') as file:
        for i in range(len(x_data)):
            file.write(str(x_data[i]) + '\t' + str(y_data[i]) + '\n')


def checkTotalEvents(spectrum):
    print('Total area:')
    s = 0
    for i in range(spectrum.channel_qty):
        s += spectrum.bin_data[i]
    print('Uncalibrated: ', s)
    if config.keep_redundant_data is True:
        s = 0
        for i in range(len(spectrum.calib_bins) - 1):
            s += (spectrum.calib_bins[i + 1] - spectrum.calib_bins[i]) * spectrum.calib_bin_data[i]
        print('Calibrated: ', s)
    s = 0
    for i in range(len(spectrum.rebin_bins) - 1):
        s += (spectrum.rebin_bins[i + 1] - spectrum.rebin_bins[i]) * spectrum.rebin_bin_data[i]
    print('Rebinned: ', s)


def prepareSet(test_set):
    counter = 0
    for spectrum in test_set:
        spectrum.rebin().calcCountRate()
        counter += 1
        print('\r', counter, '/', len(test_set), spectrum.location, end='')
    print('\n')
    return test_set


def printBinaryConfusionMatrix(srcs, bkgs, bkg_sp_reference, method, tolerance, los):
    res, tp, fp, fn, tn = None, 0, 0, 0, 0
    for spectrum in srcs:
        if not spectrum.corrupted:
            if method == 'sigma':
                res = spectrum.sigmaBinaryClassify(bkg_sp_reference)
            elif method == 'pearson':
                res = spectrum.pearsonBinaryClassify(bkg_sp_reference, tolerance=tolerance, los=los)
            elif method == 'chi2square':
                res = spectrum.chi2squareBinaryClassify(bkg_sp_reference, tolerance=tolerance, los=los)
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
                res = spectrum.chi2squareBinaryClassify(bkg_sp_reference, tolerance=tolerance, los=los)
            if res == 'Signal':
                fn += 1
            if res == 'Background':
                tn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f'\nAccuracy ({method}, {tolerance=}, {los=}): {accuracy}')
    print([tp, fp], '\n', [fn, tn])
    return [[tp, fp], [fn, tn]], accuracy


def flatten(lst):
    return [x for xs in lst for x in xs]