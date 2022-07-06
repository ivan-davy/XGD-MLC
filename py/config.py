from datetime import date
from isoclass import Isotope

test_file_location = 'sps/Cs_137_15cm.sps'
bkg_file_location = 'sps/Bkg_620s.sps'

test_fileset_location = 'sps/test'
bkg_fileset_location = 'sps/bkgs'
src_fileset_location = 'sps/sigs'


bin_clf_model_directory = 'models/bin/'
bin_clf_dataframe_directory = 'dataframes/bin/'
bin_clf_report_location = 'reports/bin_clf_report.txt'

bin_clf_sections_qty = 10
ml_bin_clf_bins_per_section = 25

bin_clf_feature_type = 'average'  # average, linfit
bin_clf_method = 'sigma'  # sigma, mlrf, , mldt, mllgr
bin_clf_only = False


clf_model_directory = 'models/clf/'
clf_dataframe_directory = 'dataframes/clf/'
clf_report_location = 'reports/clf_report.txt'

clf_feature_type = 'average'  # average, linfit
clf_method = ''


clf_isotopes = {
    'Na22': Isotope('Sodium-22', 22, 11, 2.602, 107000, 2016, 10, 1),
    'Co60': Isotope('Cobalt-60', 60, 27, 5.271, 100000, 2016, 10, 1),
    'Cs137': Isotope('Cesium-137', 137, 55, 30.05, 116000, 2016, 10, 1),
    'Am241': Isotope('Americium-241', 241, 95, 432.2, 111000, 2016, 10, 1)
}


kev_cap = 1500
sps_acquisition_date = date(2022, 7, 1)
show = False
ml_perform_data_scaling = True
keep_redundant_data = False  # False is recommended
