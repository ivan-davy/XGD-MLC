from struct import unpack_from
from config import settings
from classes.spclass import Spectrum


def loadSpectrumData(sps_filename):
    with open(sps_filename, 'rb') as file:
        buffer = file.read()
        live_time_int = (unpack_from('i', buffer, 301))[0]
        real_time_int = (unpack_from('i', buffer, 305))[0]
        num_of_bins = (unpack_from('h', buffer, 0))[0]
        calibration = unpack_from('ff', buffer, 440)
        bin_data = unpack_from(str(num_of_bins) + 'i', buffer, 1024)
    spectrum = Spectrum(sps_filename, num_of_bins, real_time_int, live_time_int, calibration, bin_data)
    if not spectrum.bin_data:
        spectrum.corrupted = True
        print(spectrum.path)
    if spectrum.cal[0] == 0:
        if settings.enforce_cal:
            spectrum.cal = settings.default_cal
        else:
            spectrum.corrupted = True
    return spectrum


def exportSpectrumData(filename, x_data, y_data):
    with open(filename, 'w') as file:
        for i in range(len(x_data)):
            file.write(str(x_data[i]) + '\t' + str(y_data[i]) + '\n')


def getSpectrumData(filename, t_norm=True, ca=None, cb=None):
    spectrum = loadSpectrumData(filename)
    if ca is not None:
        spectrum.cal[0] = ca
    if cb is not None:
        spectrum.cal[1] = cb
    spectrum.rebin()
    if not t_norm:
        return spectrum.rebin_bin_data
    if t_norm:
        spectrum.calcCountRate()
        return spectrum.count_rate_bin_data
