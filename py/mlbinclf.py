from ml import mlLoadSets, mlGetFeatures
from spclass import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import pandas as pd
import config


def mlCreateBinaryModel(source_ml_set, background_ml_set, feature_type, method,
                        bins_per_sect=ml_bin_clf_bins_per_section,
                        scale=True, show=False):
    num_of_sections = int(kev_cap / bins_per_sect)
    if feature_type == 'linfit':
        feature_names = [f'A{segment}' for segment in range(num_of_sections)] + \
                        [f'B{segment}' for segment in range(num_of_sections)]
    elif feature_type == 'average':
        feature_names = [f'C{segment}' for segment in range(num_of_sections)]
    else:
        feature_names = None
    label_names = ['Background', 'Signal']
    dframe_location = f'{config.bin_clf_dataframe_directory}{bins_per_sect}bps_{config.kev_cap}' \
                      f'keV_{feature_type}_bin.dframe'
    data_dict, model_data, label, clf = {}, None, None, None
    try:
        print(f'Looking for {dframe_location}...')
        with open(dframe_location, 'rb') as f:
            model_data = pickle.load(f)
        print('Load complete.')
    except FileNotFoundError:
        print(f'{dframe_location} not found. Creating a new one...')
        if feature_type == 'linfit':
            background_ml_set_a, background_ml_set_b = mlGetFeatures(background_ml_set,
                                                                     feature_type,
                                                                     bins_per_sect=bins_per_sect,
                                                                     show=show)
            source_ml_set_a, source_ml_set_b = mlGetFeatures(source_ml_set,
                                                             feature_type,
                                                             bins_per_sect=bins_per_sect,
                                                             show=show)
            label = [label_names[0]] * len(background_ml_set_a) + [label_names[1]] * len(source_ml_set_a)
            ml_set_a = np.array(background_ml_set_a + source_ml_set_a)
            ml_set_b = np.array(background_ml_set_b + source_ml_set_b)
            for feature in range(num_of_sections):
                data_dict[feature_names[feature]] = ml_set_a[:, feature]
            for feature in range(num_of_sections):
                data_dict[feature_names[num_of_sections + feature]] = ml_set_b[:, feature]

        elif feature_type == 'average':
            background_ml_set_c = mlGetFeatures(background_ml_set,
                                                feature_type,
                                                bins_per_sect=bins_per_sect,
                                                show=show)
            source_ml_set_c = mlGetFeatures(source_ml_set,
                                            feature_type,
                                            bins_per_sect=bins_per_sect,
                                            show=show)
            label = [label_names[0]] * len(background_ml_set_c) + [label_names[1]] * len(source_ml_set_c)
            ml_set_c = np.array(background_ml_set_c + source_ml_set_c)
            for feature in range(num_of_sections):
                data_dict[feature_names[feature]] = ml_set_c[:, feature]
        data_dict['Type'] = label
        model_data = pd.DataFrame(data_dict)
        with open(dframe_location, 'wb') as f:
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
        clf = mlFormBinaryModel(X, y, clf)
        return clf


def mlFormBinaryModel(X, y, ml_bin_model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    ml_bin_model.fit(X_train, y_train)
    y_pred = ml_bin_model.predict(X_test)
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred), '\n')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    ml_bin_model.fit(X, y)
    return ml_bin_model


def mlBinaryClassification(test_spectrum, ml_model, feature_type, bins_per_sect=ml_bin_clf_bins_per_section,
                           scale=True,
                           show=False):
    X_test = None
    if feature_type == 'linfit':
        test_ml_a, test_ml_b = mlGetFeatures([test_spectrum],
                                             feature_type,
                                             bins_per_sect=bins_per_sect,
                                             show_progress=False,
                                             show=show)
        X_test = np.array(test_ml_a[0] + test_ml_b[0]).reshape(1, -1)
    elif feature_type == 'average':
        test_ml = mlGetFeatures([test_spectrum],
                                feature_type,
                                bins_per_sect=bins_per_sect,
                                show_progress=False)
        X_test = np.array(test_ml[0]).reshape(1, -1)
    if scale:
        X_test = np.arctan(X_test)
    return ml_model.predict(X_test)[0]


def mlBinaryClassifier(test_spectrum_set, out, show, **user_args):
    import pickle
    ml_bin_model = None
    mdl_location = f'{config.bin_clf_model_directory}{"scaled_" if user_args["Scale"] else "_"}' \
                   f'{user_args["MethodBinary"]}_{user_args["FeatureBinary"]}_bin.mdl'
    try:
        print('\nLooking for the model file... ', end='')
        with open(mdl_location, 'rb') as f:
            ml_bin_model = pickle.load(f)
        print('File found.')
    except FileNotFoundError:
        print(f'{mdl_location} not found. Creating a new model...')
        sp_ml_set, bkg_ml_set = mlLoadSets()
        ml_bin_model = mlCreateBinaryModel(sp_ml_set, bkg_ml_set,
                                           user_args["FeatureBinary"],
                                           user_args["MethodBinary"],
                                           scale=user_args["Scale"],
                                           show=show)
        with open(mdl_location, 'wb') as f:
            pickle.dump(ml_bin_model, f)
        print('Done!')
    finally:
        results = {}
        for test_spectrum in test_spectrum_set:
            res = mlBinaryClassification(test_spectrum, ml_bin_model,
                                         user_args["FeatureBinary"],
                                         scale=user_args["Scale"])
            results[test_spectrum.location] = res
            out.write(f'{test_spectrum.location:<60} {mdl_location:<40} '
                      f'{user_args["MethodBinary"]:<10} {res:<10}\n')
        print(f'Binary classification results exported to {config.bin_clf_report_location}')
        return results
