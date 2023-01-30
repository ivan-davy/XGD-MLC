from pathlib import Path

test_file_location = Path('../sps/test/Сs137_15cm.sps')
bkg_file_location = Path('../sps/bkgs/60s/202275_13176_bkg_60s.sps')

test_fileset_location = Path('../sps/test')
bkg_fileset_location = Path('../sps/bkgs')
src_fileset_location = Path('../sps/srcs')

###

bin_clf_model_directory = Path('../models/bin/')
bin_clf_dataframe_directory = Path('../dataframes/bin/')
bin_clf_report_location = Path('../reports/bin_clf_report.txt')

bin_clf_sections_qty = 10
ml_bin_clf_bins_per_section = 25

bin_clf_feature_type = 'average'  # see const.const.supported_binary_clf_features
bin_clf_method = 'mlrf'  # see const.const.supported_binary_clf_methods

###

clf_model_directory = Path('../models/clf/')
clf_dataframe_directory = Path('../dataframes/clf/')
clf_report_location = Path('../reports/clf_report.txt')

ml_clf_bins_per_section = 5

clf_feature_type = 'average'  # see const.const.supported_multilabel_clf_features
clf_method = 'mllgr'  # see const.const.supported_multilabel_clf_methods

clf_display_threshold = 0.01

clf_threshold = 0.6

###

kev_cap = 1500
default_cal = [1.317870020866394, 0.1251399964094162]

enforce_cal = True
bin_clf_only = False
show = False
ml_perform_data_scaling = True  # True is recommended
delete_corrupted = True  # Recommended to set to True on first launch (backup your data!)
keep_redundant_data = False  # False is recommended
