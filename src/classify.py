from binary.mlbinclf import *
from multilabel.mlclf import *
from argparse import ArgumentParser
from os import walk, path
from utility.metrics import getClfMetrics
from utility.data import loadSpectrumData
from config import settings


def classify(**user_parsed_args):
    test_spectrum_set, counter = [], 0
    print(f'Checking for corrupted spectra files. '
          f'Note: setting delete_corrupted is set to {settings.delete_corrupted}\n.')
    for root, dirs, files in walk(user_parsed_args['TestSet']):
        for file in files:
            if file.endswith('.sps'):
                sp = loadSpectrumData(path.join(root, file))
                if not any(sp.bin_data):
                    print(f'CORRUPTED: {sp.path}')
                    if settings.delete_corrupted:
                        os.remove(sp.path)
                        continue
                test_spectrum_set.append(sp)
    print('Test files loaded:', len(test_spectrum_set), user_parsed_args['TestSet'])
    bkg_spectrum = loadSpectrumData(user_parsed_args['Bkg'])

    if test_spectrum_set:
        print('Processing test spectra...')
    for test_spectrum in test_spectrum_set:
        if test_spectrum.corrupted is False:
            test_spectrum.rebin().calcCountRate()
            counter += 1
            print('\r', counter, '/', len(test_spectrum_set), test_spectrum.path, end='')
    if bkg_spectrum.corrupted is False:
        bkg_spectrum.rebin().calcCountRate()

    print('Proceeding to binary spectra classification...')
    bin_out = open(str(user_parsed_args['OutputBinary']), 'a+')
    bin_results, res = {}, None
    if user_parsed_args['MethodBinary'] == 'sigma':
        for test_spectrum in test_spectrum_set:
            res = test_spectrum.sigmaBinaryClassify(bkg_spectrum)
            bin_results[test_spectrum.path] = res
            bin_out.write(f'{test_spectrum.path:<60} {bkg_spectrum.path:<40} '
                      f'{user_parsed_args["MethodBinary"]:<10} {res:<10}\n')
    elif 'ml' in user_parsed_args['MethodBinary']:
        bin_results = mlBinaryClassifier(test_spectrum_set,
                                         bin_out,
                                         show=False,
                                         show_progress=True,
                                         **user_parsed_args)
    else:
        print('\nRequested binary classification method not supported.')

    if not settings.bin_clf_only:
        print('Binary spectra classification completed. Proceeding to multi-label classification...')
        no_bkg_test_spectrum_set = []
        clf_out = open(str(user_parsed_args['Output']), 'a+')
        for sp in test_spectrum_set:
            if bin_results[sp.path] == 'Source':
                no_bkg_test_spectrum_set.append(sp)
        clf_results = mlClassifier(no_bkg_test_spectrum_set,
                                   clf_out,
                                   show=False,
                                   show_progress=True,
                                   show_results=False,
                                   **user_parsed_args)
        getClfMetrics(clf_results,
                      show_results=True,
                      **user_parsed_args)
    return res


if __name__ == '__main__':
    parser = ArgumentParser(description='Spectra ML Classificator')
    parser.add_argument('-T', '--TestSet',
                        help='location of spectra set to be tested',
                        default=settings.test_fileset_location,
                        type=str)
    parser.add_argument('-S', '--SigSet',
                        help='location of source spectra set',
                        default=settings.src_fileset_location,
                        type=str)
    parser.add_argument('-B', '--BkgSet',
                        help='location of background spectra set',
                        default=settings.bkg_fileset_location,
                        type=str)
    parser.add_argument('-b', '--Bkg',
                        help='location of background reference spectrum (needed for binary sigma method)',
                        default=settings.bkg_file_location,
                        type=str)
    parser.add_argument('-mb', '--MethodBinary',
                        help='binary classification method (filters out spectra with no source)',
                        default=settings.bin_clf_method,
                        type=str)
    parser.add_argument('-m', '--Method',
                        help='classification method (determines the type of sources present in a spectrum)',
                        default=settings.clf_method,
                        type=str)
    parser.add_argument('-o', '--OutputBinary',
                        help='filename/location of the output report file',
                        default=settings.bin_clf_report_location,
                        type=str)
    parser.add_argument('-O', '--Output',
                        help='filename/location of the output report file',
                        default=settings.clf_report_location,
                        type=str)
    parser.add_argument('-sc', '--Scale',
                        help='perform ML data pre-processing (boolean)',
                        default=settings.ml_perform_data_scaling,
                        type=bool)
    parser.add_argument('-F', '--Feature',
                        help='ML feature type',
                        default=settings.clf_feature_type,
                        type=str)
    parser.add_argument('-f', '--FeatureBinary',
                        help='binary ML feature type',
                        default=settings.bin_clf_feature_type,
                        type=str)
    args = vars(parser.parse_args())
    classify(**args)
