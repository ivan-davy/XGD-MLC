from mlclf.binary import *
from mlclf.multilabel import *
from argparse import ArgumentParser
from os import walk, path
from utility.metrics import getClfMetrics
from utility.data import loadSpectrumData
from config import settings
from simple_chalk import chalk
from const import const


# TODO: Отрефакторить отчеты, подобрать хорошие параметры, внедрить распознавание активности


def classify(**user_parsed_args):
    #  Loading test spectra set
    test_spectrum_set, counter = [], 0
    print(chalk.red(f'NOTE: setting delete_corrupted is set to {settings.delete_corrupted}\n'))
    print(chalk.blue(f'Loading test spectra from {user_parsed_args["TestSet"]}.'))
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
    print('Test files loaded:', chalk.cyan(len(test_spectrum_set)))

    #  Loading background reference spectrum
    bkg_spectrum = loadSpectrumData(user_parsed_args['Bkg'])

    #  Test spectra set processing
    if test_spectrum_set:
        print(chalk.blue('\nProcessing test spectra...'))
    for test_spectrum in test_spectrum_set:
        if test_spectrum.corrupted is False:
            test_spectrum.rebin().calcCountRate()
            counter += 1
            #  plotBinData(test_spectrum)
            print('\r', chalk.cyan(counter), '/', len(test_spectrum_set), test_spectrum.path, end='')
    if bkg_spectrum.corrupted is False:
        bkg_spectrum.rebin().calcCountRate()

    #  Binary spectra classification
    print(chalk.blue('\n\nProceeding to binary spectra classification...'))
    print(f'Binary classification method selected: {chalk.cyan(user_parsed_args["MethodBinary"])}')
    print(f'Binary classification feature selected: {chalk.cyan(user_parsed_args["FeatureBinary"])}')
    Path.mkdir(Path(str(user_parsed_args['OutputBinary'])).parent, exist_ok=True, parents=True)
    bin_out = open(Path(str(user_parsed_args['OutputBinary'])), 'a+')
    bin_results = {}
    if user_parsed_args["MethodBinary"] in const.supported_binary_clf_methods \
            and user_parsed_args["FeatureBinary"] in const.supported_binary_clf_features:

        # Non-ML "sigma" method
        if user_parsed_args['MethodBinary'] == 'sigma':
            print(chalk.blue('\nPerforming binary classification...'))
            for test_spectrum in test_spectrum_set:
                bin_results[test_spectrum.path] = test_spectrum.sigmaBinaryClassify(bkg_spectrum)
                bin_out.write(f'{test_spectrum.path:<100} {str(bkg_spectrum.path):<60} '
                              f'{user_parsed_args["MethodBinary"]:<15} {bin_results[test_spectrum.path]:<10}\n')
                counter += 1
                print('\r', chalk.cyan(counter), '/', len(test_spectrum_set), test_spectrum.path, end='')
            print(chalk.green(f'Binary classification results exported to {user_parsed_args["OutputBinary"]}'))

        # ML methods
        elif 'ml' in user_parsed_args['MethodBinary']:
            bin_results = mlBinaryClassifier(test_spectrum_set,
                                             bin_out,
                                             show=False,
                                             show_progress=True,
                                             **user_parsed_args)
    else:
        print(chalk.redBright('\nRequested binary classification method / feature not supported.'))
        exit()

    #  Multi-label spectra classification
    if bool_parse(user_parsed_args['NoMulti']):
        exit()
    print(chalk.blue('\nProceeding to multi-label classification...'))
    if user_parsed_args['Method'] in const.supported_multilabel_clf_methods and user_parsed_args["Feature"] \
            in const.supported_multilabel_clf_features:

        print(f'Multilabel classification method selected: {chalk.cyan(user_parsed_args["Method"])}')
        print(f'Multilabel classification feature selected: {chalk.cyan(user_parsed_args["Feature"])}\n')

        # Only-sources spectra array from binary classification
        no_bkg_test_spectrum_set = []
        clf_out = open(str(user_parsed_args['Output']), 'a+')
        for sp in test_spectrum_set:
            if bin_results[sp.path] == 'Source':
                no_bkg_test_spectrum_set.append(sp)
        clf_results = mlClassifier(no_bkg_test_spectrum_set,
                                   clf_out,
                                   show=bool_parse(user_parsed_args['Vis']),
                                   show_progress=True,
                                   show_results=bool_parse(user_parsed_args['Print']),
                                   export_images=bool_parse(user_parsed_args['Images']),
                                   **user_parsed_args)
        getClfMetrics(clf_results, **user_parsed_args)
    else:
        print(chalk.redBright('\nRequested multilabel classification method / feature not supported.'))
        exit()


if __name__ == '__main__':
    print(chalk.magenta('Starting XGD-MLC...'))
    #  CLI input parser
    parser = ArgumentParser(
        description='Xenon Gamma Detector Machine Learning Classifier -- a Python3 '
                    'ML classifier of radioactive gamma-sources, designed to work with '
                    'the spectra acquired by Xenon Gamma-Detector of NRNU MEPhI.',
        epilog=chalk.magenta('Ivan Davydov @ NRNU MEPhI, 2023'))

    parser.add_argument('-T', '--TestSet',
                        help='test spectra set location',
                        default=settings.test_fileset_dir,
                        type=str)
    parser.add_argument('-S', '--SrcSet',
                        help='sources spectra set location',
                        default=settings.src_fileset_dir,
                        type=str)
    parser.add_argument('-B', '--BkgSet',
                        help='background spectra set location',
                        default=settings.bkg_fileset_dir,
                        type=str)
    parser.add_argument('-b', '--Bkg',
                        help='background reference spectrum path (required for non-ml methods)',
                        default=settings.bkg_file_path,
                        type=str)
    parser.add_argument('-m', '--MethodBinary',
                        help='binary classification method',
                        choices=const.supported_binary_clf_methods,
                        default=settings.bin_clf_method,
                        type=str)
    parser.add_argument('-M', '--Method',
                        help='multilabel classification method',
                        choices=const.supported_multilabel_clf_methods,
                        default=settings.clf_method,
                        type=str)
    parser.add_argument('-f', '--FeatureBinary',
                        help='binary ML feature type',
                        choices=const.supported_binary_clf_features,
                        default=settings.bin_clf_feature_type,
                        type=str)
    parser.add_argument('-F', '--Feature',
                        help='multilabel ML feature type (currently one supported)',
                        choices=const.supported_multilabel_clf_features,
                        default=settings.clf_feature_type,
                        type=str)
    parser.add_argument('-o', '--OutputBinary',
                        help='binary classification report file path',
                        default=settings.bin_clf_report_path,
                        type=str)
    parser.add_argument('-O', '--Output',
                        help='multilabel classification report file path',
                        default=settings.clf_report_path,
                        type=str)

    parser.add_argument('-q', '--Scale',
                        help='perform ML data pre-processing?',
                        default=settings.ml_perform_data_scaling,
                        type=str)
    parser.add_argument('-p', '--Print',
                        help='show results?',
                        default=settings.show_results,
                        type=str)
    parser.add_argument('-i', '--Images',
                        help='export images to default image directory?',
                        default=settings.export_clf_result_images,
                        type=str)
    parser.add_argument('-V', '--Vis',
                        help='visualize multilabel classification progress',
                        default=settings.visualize_progress,
                        type=str)
    parser.add_argument('-n', '--NoMulti',
                        help='perform binary classification only',
                        default=settings.bin_clf_only,
                        type=str)

    args = vars(parser.parse_args())
    classify(**args)
