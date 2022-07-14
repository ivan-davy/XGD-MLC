from datetime import date
from isoclass import Isotope

test_file_location = 'sps/Cs_137_15cm.sps'
bkg_file_location = 'sps/bkgs/60s/202275_13333_bkg_60s.sps'

test_fileset_location = 'sps/test'
bkg_fileset_location = 'sps/bkgs'
src_fileset_location = 'sps/srcs'


bin_clf_model_directory = 'models/bin/'
bin_clf_dataframe_directory = 'dataframes/bin/'
bin_clf_report_location = 'reps/bin_clf_report.txt'

bin_clf_sections_qty = 10
ml_bin_clf_bins_per_section = 25

bin_clf_feature_type = 'average'  # average, linfit
bin_clf_method = 'mlrf'  # sigma, mlrf, mldt, mllgr


clf_model_directory = 'models/clf/'
clf_dataframe_directory = 'dataframes/clf/'
clf_report_location = 'reps/clf_report.txt'

ml_clf_bins_per_section = 25

clf_feature_type = 'average'  # average, linfit
clf_method = 'mlrf'


clf_isotopes = {
    'Na22': Isotope(0, 'Sodium-22', 22, 11, 2.602, 107000, date(2016, 10, 1), date(2022, 7, 1)),
    'Co60': Isotope(1, 'Cobalt-60', 60, 27, 5.271, 100000, date(2016, 10, 1), date(2022, 7, 1)),
    'Cs137': Isotope(2, 'Caesium-137', 137, 55, 30.05, 116000, date(2016, 10, 1), date(2022, 7, 1)),
    'Am241': Isotope(3, 'Americium-241', 241, 95, 432.2, 111000, date(2016, 10, 1), date(2022, 7, 1))
}


kev_cap = 1500
default_cal = [1.317870020866394, 0.1251399964094162]
sps_acquisition_date = date(2022, 7, 1)

enforce_cal = True
bin_clf_only = False
show = False
ml_perform_data_scaling = True

keep_redundant_data = False  # False is recommended
