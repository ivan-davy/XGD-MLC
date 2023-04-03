from pathlib import Path

test_file_path = Path('../sps/test/Сs137_15cm.sps')
bkg_file_path = Path('../sps/bkgs/60s/202275_13176_bkg_60s.sps')

test_fileset_dir = Path('../sps/test')
bkg_fileset_dir = Path('../sps/bkgs')
src_fileset_dir = Path('../sps/srcs')

images_path = Path('../images/')
temp_files_dir = Path('../temp/')

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

ml_clf_bins_per_section = 5

clf_feature_type = 'average'  # see const.const.supported_multilabel_clf_features
clf_method = 'mllgr'  # see const.const.supported_multilabel_clf_methods

clf_display_threshold = 0.01  # visual debug parameter (ignore)
clf_show_threshold = 0.2  # ~0.2 recommended
clf_threshold = 0.5  # ~0.5 recommended

###
detector_inner_r = 7

kev_cap = 1500  # no less than 1500 (or highest line kev in clf_isotopes)
default_cal = [1.317870020866394, 0.1251399964094162]  # Recommended to set manually with enforce-cal set to True
default_distance_to_src = 15
peak_delta_x = 20

enforce_cal = True  # True is recommended. Forces default_cal to all spectra, preventing NaN-issues with corrupted files

perform_filtering = False  # May negatively affect multiclf for some radionuclides, slightly better accuracy overall
filter_window = 25

perform_multi = True
predict_act = True

visualize_progress = True
show_results = False
export_clf_result_images = True

ml_perform_data_scaling = True  # True is recommended
delete_corrupted = True  # Recommended to set to True on first launch (backup your data!)
keep_redundant_data = False  # False is recommended
