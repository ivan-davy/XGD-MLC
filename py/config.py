test_file_location = 'sps/Cs_137_15cm.sps'
bkg_file_location = 'sps/bkgs/60s/202275_13333_bkg_60s.sps'

test_fileset_location = 'sps/test'
bkg_fileset_location = 'sps/bkgs'
src_fileset_location = 'sps/srcs'

###

bin_clf_model_directory = 'models/bin/'
bin_clf_dataframe_directory = 'dataframes/bin/'
bin_clf_report_location = 'reps/bin_clf_report.txt'

bin_clf_sections_qty = 10
ml_bin_clf_bins_per_section = 25

bin_clf_feature_type = 'average'  # average, linfit
bin_clf_method = 'mlrf'  # sigma, mlrf, mldt, mllgr

###

clf_model_directory = 'models/clf/'
clf_dataframe_directory = 'dataframes/clf/'
clf_report_location = 'reps/clf_report.txt'

ml_clf_bins_per_section = 5

clf_feature_type = 'average'  # average
clf_method = 'mllgr'  # mllgr, mldt, mlrf

clf_display_threshold = 0.01

clf_threshold = 0.6

###

kev_cap = 1500
default_cal = [1.317870020866394, 0.1251399964094162]

enforce_cal = True
bin_clf_only = False
show = False
ml_perform_data_scaling = True
delete_corrupted = True  # Recommended to set to True on first launch (backup your data!)
keep_redundant_data = False  # False is recommended
