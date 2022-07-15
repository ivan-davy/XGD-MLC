import config
from visual import *
from mlbinclf import *
from mlclf import *
from argparse import ArgumentParser
from cProfile import Profile
from os import walk, path
from utility import loadSpectrumData


def classify(**user_parsed_args):
    with Profile() as pr:
        test_spectrum_set, counter = [], 0
        print(f'Checking for corrupted spectra files. Setting delete_corrupted is set to {config.delete_corrupted}\n')
        for root, dirs, files in walk(user_parsed_args['TestSet']):
            for file in files:
                if file.endswith('.sps'):
                    sp = loadSpectrumData(path.join(root, file))
                    if not any(sp.bin_data):
                        print(f'CORRUPTED: {sp.location}')
                        if config.delete_corrupted:
                            os.remove(sp.location)
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
                print('\r', counter, '/', len(test_spectrum_set), test_spectrum.location, end='')
        if bkg_spectrum.corrupted is False:
            bkg_spectrum.rebin().calcCountRate()

        bin_out = open(str(user_parsed_args['OutputBinary']), 'a+')
        bin_results, res = {}, None
        if user_parsed_args['MethodBinary'] == 'sigma':
            for test_spectrum in test_spectrum_set:
                res = test_spectrum.sigmaBinaryClassify(bkg_spectrum)
                bin_results[test_spectrum.location] = res
                bin_out.write(f'{test_spectrum.location:<60} {bkg_spectrum.location:<40} '
                          f'{user_parsed_args["MethodBinary"]:<10} {res:<10}\n')
        elif 'ml' in user_parsed_args['MethodBinary']:
            bin_results = mlBinaryClassifier(test_spectrum_set,
                                             bin_out,
                                             show=False,
                                             show_results=True,
                                             **user_parsed_args)
        else:
            print('\nRequested binary classification method not supported.')

        if not config.bin_clf_only:
            print('Binary spectra classification completed. Proceeding to multi-label classification...')
            no_bkg_test_spectrum_set = []
            clf_out = open(str(user_parsed_args['Output']), 'a+')
            for sp in test_spectrum_set:
                if bin_results[sp.location] == 'Source':
                    no_bkg_test_spectrum_set.append(sp)
            clf_results = mlClassifier(no_bkg_test_spectrum_set, clf_out, show=False, **user_parsed_args)


    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    return res


def getSpectrumData(filename, t_norm=True, ca=None, cb=None):
    spectrum = loadSpectrumData(filename)
    if ca is not None:
        spectrum.cal[0] = ca
    if cb is not None:
        spectrum.cal[1] = cb
    spectrum.rebin()
    if not t_norm:
        return spectrum.rebin_bin_data
    if t_norm:
        spectrum.calcCountRate()
        return spectrum.count_bin_data


if __name__ == '__main__':
    parser = ArgumentParser(description='Spectra ML Classificator')
    parser.add_argument('-T', '--TestSet',
                        help='location of spectra set to be tested',
                        default=config.test_fileset_location,
                        type=str)
    parser.add_argument('-S', '--SigSet',
                        help='location of source spectra set',
                        default=config.src_fileset_location,
                        type=str)
    parser.add_argument('-B', '--BkgSet',
                        help='location of background spectra set',
                        default=config.bkg_fileset_location,
                        type=str)
    parser.add_argument('-b', '--Bkg',
                        help='location of background reference spectrum (needed for binary sigma method)',
                        default=config.bkg_file_location,
                        type=str)
    parser.add_argument('-mb', '--MethodBinary',
                        help='binary classification method (filters out spectra with no source)',
                        default=config.bin_clf_method,
                        type=str)
    parser.add_argument('-m', '--Method',
                        help='classification method (determines the type of sources present in a spectrum)',
                        default=config.clf_method,
                        type=str)
    parser.add_argument('-o', '--OutputBinary',
                        help='filename/location of the output report file',
                        default=config.bin_clf_report_location,
                        type=str)
    parser.add_argument('-O', '--Output',
                        help='filename/location of the output report file',
                        default=config.clf_report_location,
                        type=str)
    parser.add_argument('-sc', '--Scale',
                        help='perform ML data pre-processing (boolean)',
                        default=config.ml_perform_data_scaling,
                        type=bool)
    parser.add_argument('-F', '--Feature',
                        help='ML feature type',
                        default=config.clf_feature_type,
                        type=str)
    parser.add_argument('-f', '--FeatureBinary',
                        help='binary ML feature type',
                        default=config.bin_clf_feature_type,
                        type=str)
    args = vars(parser.parse_args())
    classify(**args)