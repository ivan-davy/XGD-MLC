from pathlib import Path

test_file_path = Path('../sps/test/Сs137_15cm.sps')
bkg_file_path = Path('../sps/bkgs/60s/202275_13176_bkg_60s.sps')

test_fileset_dir = Path('../sps/test')
bkg_fileset_dir = Path('../sps/bkgs')
src_fileset_dir = Path('../sps/srcs')

###

bin_clf_model_dir = Path('../models/bin/')
bin_clf_dataframe_dir = Path('../dataframes/bin/')
bin_clf_report_path = Path('../reports/bin_clf_report.txt')

bin_clf_sections_qty = 10
ml_bin_clf_bins_per_section = 25

bin_clf_feature_type = 'average'  # see const.const.supported_binary_clf_features
bin_clf_method = 'mlrf'  # see const.const.supported_binary_clf_methods

###

clf_model_dir = Path('../models/multi/')
clf_dataframe_dir = Path('../dataframes/multi/')
clf_report_path = Path('../reports/clf_report.txt')
clf_images_path = Path('../images/')

ml_clf_bins_per_section = 5

clf_feature_type = 'average'  # see const.const.supported_multilabel_clf_features
clf_method = 'mllgr'  # see const.const.supported_multilabel_clf_methods

clf_display_threshold = 0.01

clf_threshold = 0.6

###

kev_cap = 1500
default_cal = [1.317870020866394, 0.1251399964094162]  # Recommended to set manually with enforce-cal set to True

enforce_cal = True  # True is recommended. Forces default_cal to all spectra, preventing NaN-issues with corrupted files
bin_clf_only = False
show_results = False
export_images = True

ml_perform_data_scaling = True  # True is recommended
delete_corrupted = True  # Recommended to set to True on first launch (backup your data!)
keep_redundant_data = False  # False is recommended
