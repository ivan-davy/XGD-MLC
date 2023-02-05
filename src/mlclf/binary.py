import os
from classes.spclass import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import pandas as pd
from config import settings
import numpy as np
from utility.data import loadSpectrumData
from utility.visual import mlShowLinfit, mlShowAverage
from simple_chalk import chalk


def mlLoadBinarySets():
    from os import walk, path
    source_sp_set, background_sp_set = [], []
    for root, dirs, files in walk(settings.src_fileset_location):
        for file in files:
            if file.endswith('.sps'):
                sp = loadSpectrumData(path.join(root, file))
                if not any(sp.bin_data):
                    print(chalk.redBright(f'CORRUPTED: {sp.path}'))
                    if settings.delete_corrupted:
                        os.remove(sp.path)
                        continue
                sp.has_source = True
                source_sp_set.append(sp)
    for root, dirs, files in walk(settings.bkg_fileset_location):
        for file in files:
            if file.endswith('.sps'):
                sp = loadSpectrumData(path.join(root, file))
                if not any(sp.bin_data):
                    print(chalk.redBright(f'CORRUPTED: {sp.path}'))
                    if settings.delete_corrupted:
                        os.remove(sp.path)
                        continue
                sp.has_source = False
                background_sp_set.append(sp)
    return source_sp_set, background_sp_set


def mlGetBinaryFeatures(ml_set, feature_type, bins_per_sect=settings.ml_bin_clf_bins_per_section,
                        show_progress=True,
                        show=False):
    num_of_sections = int(settings.kev_cap / bins_per_sect)
    if feature_type == 'linfit':
        from utility.visual import linear
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
                                       spectrum.count_rate_bin_data[section * bins_per_sect:(section + 1) * bins_per_sect])
                spectrum_a.append(section_ab[0][0])
                spectrum_b.append(section_ab[0][1])
            if show:
                mlShowLinfit(spectrum, spectrum_a, spectrum_b, bins_per_sect)
            set_a.append(spectrum_a)
            set_b.append(spectrum_b)
            if show_progress:
                print('\r', chalk.cyan(counter), '/', len(ml_set), spectrum.path, end='')
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
                section_c = spectrum.count_rate_bin_data[section * bins_per_sect:(section + 1) * bins_per_sect]
                spectrum_c.append(sum(section_c) / bins_per_sect)
            if show:
                mlShowAverage(spectrum, spectrum_c, bins_per_sect)
            set_c.append(spectrum_c)
            if show_progress:
                print('\r', chalk.cyan(counter), '/', len(ml_set), spectrum.path, end='')
        if show_progress:
            print('\n')
        return set_c


def mlCreateBinaryModel(source_ml_set, background_ml_set, feature_type, method,
                        bins_per_sect=settings.ml_bin_clf_bins_per_section,
                        scale=True, show=False):
    num_of_sections = int(settings.kev_cap / bins_per_sect)
    if feature_type == 'linfit':
        feature_names = [f'A{segment}' for segment in range(num_of_sections)] + \
                        [f'B{segment}' for segment in range(num_of_sections)]
    elif feature_type == 'average':
        feature_names = [f'C{segment}' for segment in range(num_of_sections)]
    else:
        feature_names = None
    label_names = ['Background', 'Source']
    dframe_location = f'{settings.bin_clf_dataframe_directory}{os.sep}{bins_per_sect}bps_{settings.kev_cap}' \
                      f'keV_{feature_type}_bin.dfr'
    data_dict, model_data, labels, clf = {}, None, None, None
    try:
        print(chalk.blue(f'Looking for {dframe_location}...'))
        with open(dframe_location, 'rb') as f:
            model_data = pickle.load(f)
        print(chalk.green('Load complete.'))
    except FileNotFoundError:
        print(chalk.red(f'Dataframe file not found.'), 'Creating a new one...\n')
        if feature_type == 'linfit':
            background_ml_set_a, background_ml_set_b = mlGetBinaryFeatures(background_ml_set,
                                                                           feature_type,
                                                                           bins_per_sect=bins_per_sect,
                                                                           show=show)
            source_ml_set_a, source_ml_set_b = mlGetBinaryFeatures(source_ml_set,
                                                                   feature_type,
                                                                   bins_per_sect=bins_per_sect,
                                                                   show=show)
            labels = [label_names[0]] * len(background_ml_set_a) + [label_names[1]] * len(source_ml_set_a)
            ml_set_a = np.array(background_ml_set_a + source_ml_set_a)
            ml_set_b = np.array(background_ml_set_b + source_ml_set_b)
            for feature in range(num_of_sections):
                data_dict[feature_names[feature]] = ml_set_a[:, feature]
            for feature in range(num_of_sections):
                data_dict[feature_names[num_of_sections + feature]] = ml_set_b[:, feature]

        elif feature_type == 'average':
            background_ml_set_c = mlGetBinaryFeatures(background_ml_set,
                                                      feature_type,
                                                      bins_per_sect=bins_per_sect,
                                                      show=show)
            source_ml_set_c = mlGetBinaryFeatures(source_ml_set,
                                                  feature_type,
                                                  bins_per_sect=bins_per_sect,
                                                  show=show)
            labels = [label_names[0]] * len(background_ml_set_c) + [label_names[1]] * len(source_ml_set_c)
            ml_set_c = np.array(background_ml_set_c + source_ml_set_c)
            for feature in range(num_of_sections):
                data_dict[feature_names[feature]] = ml_set_c[:, feature]
        data_dict['Type'] = labels
        model_data = pd.DataFrame(data_dict)
        with open(dframe_location, 'wb+') as f:
            pickle.dump(model_data, f)
    finally:
        if method == 'mlrf':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        elif method == 'mldt':
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier()
        elif method == 'mllgr':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
        X = model_data[feature_names].values
        if scale:
            X = np.arctan(X)
        y = model_data['Type']
        clf = mlFormCompleteBinaryModel(X, y, clf)
        return clf


def mlFormCompleteBinaryModel(X, y, ml_bin_model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    ml_bin_model.fit(X_train, y_train)
    y_pred = ml_bin_model.predict(X_test)
    print(chalk.cyan('Confusion matrix:\n'), confusion_matrix(y_test, y_pred), '\n')
    print(chalk.cyan('Accuracy:'), accuracy_score(y_test, y_pred))
    ml_bin_model.fit(X, y)
    return ml_bin_model


def mlBinaryClassification(test_spectrum, ml_model, feature_type, bins_per_sect=settings.ml_bin_clf_bins_per_section,
                           scale=True, show=False):
    X_test = None
    if feature_type == 'linfit':
        test_ml_a, test_ml_b = mlGetBinaryFeatures([test_spectrum],
                                                   feature_type,
                                                   bins_per_sect=bins_per_sect,
                                                   show_progress=False,
                                                   show=show)
        X_test = np.array(test_ml_a[0] + test_ml_b[0]).reshape(1, -1)
    elif feature_type == 'average':
        test_ml = mlGetBinaryFeatures([test_spectrum],
                                      feature_type,
                                      bins_per_sect=bins_per_sect,
                                      show_progress=False)
        X_test = np.array(test_ml[0]).reshape(1, -1)
    if scale:
        X_test = np.arctan(X_test)
    return ml_model.predict(X_test)[0]


def mlBinaryClassifier(test_spectrum_set, out, show, show_progress, **user_args):
    import pickle
    ml_bin_model = None
    mdl_location = f'{settings.bin_clf_model_directory}{os.sep}{settings.ml_bin_clf_bins_per_section}bps_{settings.kev_cap}_kev' \
                   f'{"_scaled_" if user_args["Scale"] else "_"}' \
                   f'{user_args["MethodBinary"]}_{user_args["FeatureBinary"]}_bin.mdl'
    try:
        print(chalk.blue(f'\nLooking for {mdl_location}... '), end='')
        with open(mdl_location, 'rb') as f:
            ml_bin_model = pickle.load(f)
        print(chalk.green('File found.'))
    except FileNotFoundError:
        print(chalk.red(f'\nModel file not found. '), 'Creating a new one...')
        sp_ml_set, bkg_ml_set = mlLoadBinarySets()
        ml_bin_model = mlCreateBinaryModel(sp_ml_set, bkg_ml_set,
                                           user_args["FeatureBinary"],
                                           user_args["MethodBinary"],
                                           scale=user_args["Scale"],
                                           show=show)
        with open(mdl_location, 'wb') as f:
            pickle.dump(ml_bin_model, f)
        print(chalk.green('Done!'))
    finally:
        print(chalk.blue('\nPerforming binary classification...'))
        results, counter = {}, 0
        for test_spectrum in test_spectrum_set:
            if show_progress:
                counter += 1
                print('\r', chalk.cyan(counter), '/', len(test_spectrum_set), test_spectrum.path, end='')
            res = mlBinaryClassification(test_spectrum, ml_bin_model,
                                         user_args["FeatureBinary"],
                                         scale=user_args["Scale"])
            results[test_spectrum.path] = res
            out.write(f'{test_spectrum.path:<100} {mdl_location:<40} '
                      f'{user_args["MethodBinary"]:<15} {res:<10}\n')
        print(chalk.green(f'\nBinary classification results exported to {settings.bin_clf_report_location}'))
        return results
