from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import config
import pickle
import pandas as pd
import numpy as np
from visual import mlShowLinfit, mlShowAverage
from utility import loadSpectrumData, flatten


def mlLoadSets():
    from os import walk, path, scandir
    sp_set = {}
    subdirs = [f.path for f in scandir(f'{config.src_fileset_location}') if f.is_dir()]
    for subdir in subdirs:
        for root, dirs, files in walk(subdir):
            for file in files:
                if file.endswith('.sps'):
                    dir_name = subdir.rsplit('/', 1)[-1]
                    if config.test_fileset_location not in dir_name:
                        sp = loadSpectrumData(path.join(root, file))
                        known_isotope = config.clf_isotopes[dir_name]
                        sp.isotope = known_isotope
                        sp_set[sp.location] = sp
    return sp_set


def mlGetFeatures(spectrum, feature_type, bins_per_sect=config.ml_bin_clf_bins_per_section,
                  show_progress=True,
                  show=False):
    num_of_sections = int(config.kev_cap / bins_per_sect)
    if feature_type == 'linfit':
        from visual import linear
        from scipy.optimize import curve_fit
        # set_a, set_b, counter = [], [], 0
        # for spectrum in ml_set:
        #     counter += 1
        #     spectrum_a, spectrum_b = [], []
        #     if spectrum.corrupted is False:
        #         spectrum.rebin()
        #         spectrum.calcCountRate()
        #     else:
        #         continue
        #     for section in range(num_of_sections):
        #         section_ab = curve_fit(linear,
        #                                spectrum.rebin_bins[section * bins_per_sect:(section + 1) * bins_per_sect],
        #                                spectrum.count_bin_data[section * bins_per_sect:(section + 1) * bins_per_sect])
        #         spectrum_a.append(section_ab[0][0])
        #         spectrum_b.append(section_ab[0][1])
        #     if show:
        #         mlShowLinfit(spectrum, spectrum_a, spectrum_b, bins_per_sect)
        #     set_a.append(spectrum_a)
        #     set_b.append(spectrum_b)
        #     if show_progress:
        #         print('\r', counter, '/', len(ml_set), spectrum.location, end='')
        # if show_progress:
        #     print('\n')
        # return set_a, set_b

    elif feature_type == 'average':
        spectrum_c = []
        if spectrum.corrupted is False:
            spectrum.rebin()
            spectrum.calcCountRate()
            for section in range(num_of_sections):
                section_c = spectrum.count_bin_data[section * bins_per_sect:(section + 1) * bins_per_sect]
                spectrum_c.append(sum(section_c) / bins_per_sect)
            if show:
                mlShowAverage(spectrum, spectrum_c, bins_per_sect)
        return spectrum_c


def mlCreateModel(sp_set, feature_type, method,
                  bins_per_sect=config.ml_clf_bins_per_section,
                  scale=True, show=False, show_progress=True):
    num_of_sections = int(config.kev_cap / bins_per_sect)
    if feature_type == 'linfit':
        feature_names = [f'A{segment}' for segment in range(num_of_sections)] + \
                        [f'B{segment}' for segment in range(num_of_sections)]
    elif feature_type == 'average':
        feature_names = [f'C{segment}' for segment in range(num_of_sections)]
    else:
        feature_names = None

    label_arrays = []
    for key, value in config.clf_isotopes.items():
        iso_list = [0] * len(config.clf_isotopes)
        iso_list[value.label] = 1
        label_arrays.append(iso_list)

    dframe_location = f'{config.clf_dataframe_directory}{bins_per_sect}bps_{config.kev_cap}' \
                      f'keV_{feature_type}.dframe'
    data_dict, model_data, labels, clf = {}, None, None, None
    try:
        print(f'Looking for {dframe_location}...')
        with open(dframe_location, 'rb') as f:
            data = pickle.load(f)
            dataframe = data['dataframe']
            y = data['labels']
        print('Load complete.')
    except FileNotFoundError:
        print(f'{dframe_location} not found. Creating a new one...')
        if feature_type == 'linfit':
            ml_sets_a, ml_sets_b = {}, {}
            for key, value in sp_set.items():
                ml_sets_a[key], ml_sets_b[key] = mlGetFeatures(value,
                                                               feature_type,
                                                               bins_per_sect=bins_per_sect,
                                                               show=show)
            labels = []
            for key, value in ml_sets_a.items():
                labels.append([label_arrays[value.label]] * len(value))
            labels = flatten(labels)
            sum_list = []
            for key, value in ml_sets_a.items():
                sum_list += value
            ml_set_a = np.array(ml_sets_a)
            sum_list = []
            for key, value in ml_sets_b.items():
                sum_list += value
            ml_set_b = np.array(ml_sets_b)

            for feature in range(num_of_sections):
                data_dict[feature_names[feature]] = ml_set_a[:, feature]
            for feature in range(num_of_sections):
                data_dict[feature_names[num_of_sections + feature]] = ml_set_b[:, feature]
        elif feature_type == 'average':
            counter, y = 0, []
            data_features_set = {}
            for key, value in sp_set.items():
                value.features_array = mlGetFeatures(value,
                                                     feature_type,
                                                     bins_per_sect=bins_per_sect,
                                                     show=show)
                data_features_set[key] = value.features_array
                y.append(value.isotope.name)
                counter += 1
                if show_progress:
                    print('\r', counter, '/', len(sp_set), key, end='')
            dataframe = pd.DataFrame.from_dict(data_features_set, orient='index', columns=labels)
            print('\n', dataframe)
        with open(dframe_location, 'wb+') as f:
            pickle.dump({
                'dataframe': dataframe,
                'labels': y
            }, f)
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
        X = dataframe.values
        if scale:
            X = np.arctan(X)
        clf = mlFormModel(X, y, clf)
        return clf


def mlFormModel(X, y, ml_clf_model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    ml_clf_model.fit(X_train, y_train)
    y_pred = ml_clf_model.predict(X_test)
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred), '\n')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    ml_clf_model.fit(X, y)
    return ml_clf_model


def mlClassification(test_spectrum, ml_model, feature_type, bins_per_sect=config.ml_clf_bins_per_section,
                     scale=True,
                     show=False):
    X_test = None
    if feature_type == 'linfit':
        test_ml_a, test_ml_b = mlGetFeatures(test_spectrum,
                                             feature_type,
                                             bins_per_sect=bins_per_sect,
                                             show_progress=False,
                                             show=show)
        X_test = np.array(test_ml_a[0] + test_ml_b[0]).reshape(1, -1)
    elif feature_type == 'average':
        test_ml = mlGetFeatures(test_spectrum,
                                feature_type,
                                bins_per_sect=bins_per_sect,
                                show_progress=False)
        X_test = np.array(test_ml).reshape(1, -1)
    if scale:
        X_test = np.arctan(X_test)
    print(X_test)
    return ml_model.predict(X_test)[0]


def mlClassifier(test_spectrum_set, out, show, **user_args):
    import pickle
    ml_clf_model = None
    mdl_location = f'{config.clf_model_directory}{"scaled_" if user_args["Scale"] else "_"}' \
                   f'{user_args["Method"]}_{user_args["Feature"]}_clf.mdl'
    try:
        print(f'\nLooking for {mdl_location}... ', end='')
        with open(mdl_location, 'rb') as f:
            ml_clf_model = pickle.load(f)
        print('File found.')
    except FileNotFoundError:
        print(f'Model file not found. Creating a new one...')
        sp_set = mlLoadSets()
        ml_clf_model = mlCreateModel(sp_set,
                                     user_args["Feature"],
                                     user_args["Method"],
                                     scale=user_args["Scale"],
                                     show=show,
                                     show_progress=True)
        with open(mdl_location, 'wb+') as f:
            pickle.dump(ml_clf_model, f)
        print('Done!')
    finally:
        results = {}
        for test_spectrum in test_spectrum_set:
            res = mlClassification(test_spectrum, ml_clf_model,
                                   user_args["Feature"],
                                   scale=user_args["Scale"])
            results[test_spectrum.location] = res
            out.write(f'{test_spectrum.location:<60} '
                      f'{user_args["Method"]:<10} {res:<10}\n')
        print(f' classification results exported to {config.clf_report_location}')
        return results
