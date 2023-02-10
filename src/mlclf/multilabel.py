import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
from config import isodata, settings
from utility.visual import mlShowAverage, plotClassificationResults
from utility.data import loadSpectrumData
from utility.common import bool_parse
import os
from simple_chalk import chalk
from activity import activity


def mlLoadSets(user_args):
    from os import walk, path, scandir
    sp_set = {}
    subdirs = [f.path for f in scandir(f'{user_args["SrcSet"]}') if f.is_dir()]
    for subdir in subdirs:
        for root, dirs, files in walk(subdir):
            for file in files:
                if file.endswith('.sps'):
                    dir_name = subdir.rsplit(os.sep, 1)[-1]
                    if str(user_args["TestSet"]) not in dir_name:
                        sp = loadSpectrumData(path.join(root, file))
                        if not any(sp.bin_data):
                            print(chalk.redBright(f'CORRUPTED: {sp.path}'))
                            if settings.delete_corrupted:
                                os.remove(sp.path)
                                continue
                        sp.src_known_isotope = isodata.clf_isotopes[dir_name]
                        sp_set[sp.path] = sp
    return sp_set


def mlGetFeatures(spectrum, feature_type, bins_per_sect=settings.ml_bin_clf_bins_per_section, show=False):
    num_of_sections = int(settings.kev_cap / bins_per_sect)
    if feature_type == 'average':
        spectrum_c = []
        if spectrum.corrupted is False:
            spectrum.rebin()
            spectrum.calcCountRate()
            for section in range(num_of_sections):
                section_c = spectrum.count_rate_bin_data[section * bins_per_sect:(section + 1) * bins_per_sect]
                spectrum_c.append(sum(section_c) / bins_per_sect)
            if show:
                mlShowAverage(spectrum, spectrum_c, bins_per_sect)
        return spectrum_c


def mlCreateModel(sp_set,
                  feature_type,
                  method,
                  bins_per_sect=settings.ml_clf_bins_per_section,
                  scale=True,
                  show=False,
                  show_progress=True):
    num_of_sections = int(settings.kev_cap / bins_per_sect)
    feature_names = None
    if feature_type == 'average':
        feature_names = [f'C{segment}' for segment in range(num_of_sections)]

    dframe_location = f'{settings.clf_dataframe_dir}{os.sep}{bins_per_sect}bps_{settings.kev_cap}' \
                      f'keV_{feature_type}_multi.dfr'
    data_dict, dataframe, y, model_data, labels, clf = {}, None, None, None, None, None
    try:
        print(chalk.blue(f'Looking for {dframe_location}...'))
        with open(dframe_location, 'rb') as f:
            data = pickle.load(f)
            dataframe = data['dataframe']
            y = data['labels']
        print(chalk.green('Load complete.'))
    except FileNotFoundError:
        print(chalk.yellow(f'Dataframe file not found. '), 'Creating a new one...\n')
        if feature_type == 'average':
            counter, y = 0, []
            data_features_set = {}
            for key, value in sp_set.items():
                value.features_array = mlGetFeatures(value,
                                                     feature_type,
                                                     bins_per_sect=bins_per_sect,
                                                     show=show)
                data_features_set[key] = value.features_array
                y.append(value.src_known_isotope.name)
                if show_progress:
                    counter += 1
                    print('\r', chalk.cyan(counter), '/', len(sp_set), key, end='')
            dataframe = pd.DataFrame.from_dict(data_features_set, orient='index', columns=feature_names)
        os.makedirs(os.path.dirname(dframe_location), exist_ok=True)
        with open(dframe_location, 'wb+') as f:
            pickle.dump({
                'dataframe': dataframe,
                'labels': y
            }, f)
    finally:
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.linear_model import LogisticRegression
        binary_indicator_arrays = {}
        for key, value in isodata.clf_isotopes.items():
            iso_list = [0] * len(isodata.clf_isotopes)
            iso_list[value.iso_id] = 1
            binary_indicator_arrays[value.name] = iso_list
        y_bin = []
        for label in y:
            y_bin.append(binary_indicator_arrays[label])
            # print(label, binary_indicator_arrays[label])
        if method == 'mlrf':
            from sklearn.ensemble import RandomForestClassifier
            clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=500, n_jobs=-1))
        elif method == 'mldt':
            from sklearn.tree import DecisionTreeClassifier
            clf = MultiOutputClassifier(DecisionTreeClassifier())
        elif method == 'mllgr':
            from sklearn.linear_model import LogisticRegression
            clf = MultiOutputClassifier(LogisticRegression())
        X = np.array(dataframe.values)
        if scale:
            X = np.arctan(X)
        clf = mlFormCompleteModel(X, y_bin, clf)
        return clf


def mlFormCompleteModel(X, y, ml_clf_model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    ml_clf_model.fit(X_train, y_train)
    y_pred = ml_clf_model.predict(X_test)
    print('\nPreliminary model accuracy:', chalk.cyan(accuracy_score(y_test, y_pred)))
    ml_clf_model.fit(X, y)
    return ml_clf_model


def mlClassification(test_spectrum, ml_model, feature_type, bins_per_sect=settings.ml_clf_bins_per_section,
                     scale=True):
    X_test = None
    if feature_type == 'average':
        test_ml = mlGetFeatures(test_spectrum,
                                feature_type,
                                bins_per_sect=bins_per_sect)
        X_test = np.array(test_ml).reshape(1, -1)
    if scale:
        X_test = np.arctan(X_test)
    res_proba = np.array(ml_model.predict_proba(X_test)).reshape(-1)[1::2]
    return res_proba


def mlClassifier(test_spectrum_set, out, predict_act=True,
                 show=False, show_progress=True, show_results=True, export_images=True, **user_args):
    import pickle
    ml_clf_model = None
    mdl_location = f'{settings.clf_model_dir}{os.sep}' \
                   f'{settings.ml_clf_bins_per_section}bps_{settings.kev_cap}kev' \
                   f'{"_scaled_" if bool_parse(user_args["Scale"]) else "_"}' \
                   f'{user_args["Method"]}_{user_args["Feature"]}_multi.mdl'
    try:
        print(chalk.blue(f'Looking for {mdl_location}... '), end='')
        with open(mdl_location, 'rb') as f:
            ml_clf_model = pickle.load(f)
        print(chalk.green('File found.'))
    except FileNotFoundError:
        print(chalk.yellow(f'\nModel file not found. '), 'Creating a new one...')
        sp_set = mlLoadSets(user_args)
        ml_clf_model = mlCreateModel(sp_set,
                                     user_args["Feature"],
                                     user_args["Method"],
                                     scale=bool_parse(user_args["Scale"]),
                                     show=False,
                                     show_progress=show_progress)
        os.makedirs(os.path.dirname(mdl_location), exist_ok=True)
        with open(mdl_location, 'wb+') as f:
            pickle.dump(ml_clf_model, f)
        print(chalk.green('Done!'))
    finally:
        print(chalk.blue('\nPerforming multilabel classification...'))
        results_proba, results_act, counter = {}, {}, 0
        app, vis = None, None
        if show:
            from PyQt5.QtWidgets import QApplication
            from classes.visualizer import Visualizer
            app = QApplication(sys.argv)
            vis = Visualizer()

        for test_spectrum in test_spectrum_set:
            if show_progress:
                counter += 1
                print('\r', chalk.cyan(counter), '/', len(test_spectrum_set), test_spectrum.path, end='')

            #  Classification + probability predictions
            test_spectrum_proba_result, i = {}, 0
            res_proba = mlClassification(test_spectrum, ml_clf_model, user_args["Feature"], scale=user_args["Scale"])

            for key, value in isodata.clf_isotopes.items():
                custom_proba = res_proba[i] * isodata.clf_proba_custom_multipliers[key]
                if custom_proba > 1:
                    custom_proba = 1
                if custom_proba < settings.clf_show_threshold:
                    i += 1
                    continue
                test_spectrum_proba_result[key] = round(custom_proba, 3)
                i += 1

            results_proba[test_spectrum.path] = test_spectrum_proba_result
            sorted_result_keys = sorted(test_spectrum_proba_result, key=test_spectrum_proba_result.get, reverse=True)
            test_spectrum_proba_result_sorted = {}
            for w in sorted_result_keys:
                test_spectrum_proba_result_sorted[w] = test_spectrum_proba_result[w]

            test_spectrum_act_result = {key: None for key in test_spectrum_proba_result_sorted.keys()}
            if predict_act:
                isotope_names_to_predict = test_spectrum_proba_result_sorted.keys()
                test_spectrum_act_result = activity.predictActivity(test_spectrum,
                                                                    isotope_names_to_predict)

            os.makedirs(os.path.dirname(user_args['Output']), exist_ok=True)
            settings.images_path.mkdir(exist_ok=True, parents=True)
            out.write(f'{test_spectrum.path:<100} '
                      f'{settings.ml_clf_bins_per_section:<3}bps '
                      f'{user_args["Method"]:<15}'
                      f'{test_spectrum_proba_result_sorted}\n')

            if show_results or export_images:
                plotClassificationResults(test_spectrum,
                                          test_spectrum_proba_result_sorted,
                                          test_spectrum_act_result,
                                          show_results=show_results,
                                          show=show,
                                          export=export_images,
                                          vis=vis)
        print(chalk.green(f'\nClassification results exported to {user_args["Output"]}'))
        if export_images:
            print(chalk.green(f'Images exported to {settings.images_path}'))

        if show:
            app.quit()

        return results_proba, results_act
