import math
import numpy as np
import scipy.special
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
from scipy import optimize
from config import settings, isodata
from utility.common import linear


def plotAllBinData(spectrum):
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle(spectrum.path)
    axs[0, 0].set(xlabel='ADC bins', ylabel='Events per bin')
    x_bin = np.arange(0, spectrum.channel_qty)
    axs[0, 0].step(x_bin, spectrum.bin_data, color='red')
    axs[0, 0].set_xlim([0, spectrum.channel_qty / 2])
    axs[0, 1].set(xlabel='keV', ylabel='Events per bin')
    axs[0, 1].step(spectrum.calib_bins, spectrum.calib_bin_data, color='blue')
    axs[0, 1].set_xlim([0, 2000])
    axs[1, 0].set(xlabel='keV (Rebinned)', ylabel='Count rate per keV')
    axs[1, 0].step(spectrum.rebin_bins, spectrum.count_rate_bin_data, color='purple')
    axs[1, 0].set_xlim([0, 2000])
    axs[1, 1].set(xlabel='keV (Rebinned)', ylabel='Events per keV')
    axs[1, 1].step(spectrum.rebin_bins, spectrum.rebin_bin_data, color='black')
    axs[1, 1].set_xlim([0, 2000])
    fig.canvas.manager.full_screen_toggle()
    plt.show()


GAUSS_CUSTOM_PARAMETERS = ('A', 'B', 'Amp', 'Mu', 'Sigma')


def gaussCustom(x, a, b, amp, mu, sig):
    return a * x + b + amp / (math.sqrt(2 * math.pi) * sig) * math.exp(-0.5 * ((x - mu) / sig) ** 2)


def fitGaussCustom(x_data, y_data, x_range, guess, to_plot):
    x, y = [], []
    for i in range(len(x_data) - 1):
        if x_range[0] < x_data[i] < x_range[1]:
            x.append(x_data[i])
            y.append(y_data[i])
    params, cov, *_ = optimize.curve_fit(np.vectorize(gaussCustom), x, y, p0=guess)
    print('Fit Parameters:')
    for p in range(len(params) - 1):
        print(params[p], ' +/- ', cov[p][p])
    if to_plot:
        plotGaussCustom(x, y, params)
    return params, cov


def plotGaussCustom(x, y, params):
    a, b, amp, mu, sig = params[0], params[1], params[2], params[3], params[4]
    fit_data = [gaussCustom(x[i], a, b, amp, mu, sig) for i in range(len(x))]
    fit_data_linear = [a * x[i] + b for i in range(len(x))]
    plt.plot(x, y, 'o', markersize=2, label='Data')
    plt.plot(x, fit_data, '-', label='Fitted')
    plt.plot(x, fit_data_linear, '-', color='cyan', label='Base')
    plt.legend()
    plt.show()


def fitAndPlot2GaussCustom(x_data1, y_data1, x_range1, guess1, x_data2, y_data2, x_range2, guess2):
    x1, y1 = [], []
    for i in range(len(x_data1) - 1):
        if x_range1[0] < x_data1[i] < x_range1[1]:
            x1.append(x_data1[i])
            y1.append(y_data1[i])
    x2, y2 = [], []
    for i in range(len(x_data2) - 1):
        if x_range2[0] < x_data2[i] < x_range2[1]:
            x2.append(x_data2[i])
            y2.append(y_data2[i])
    params1, cov1, *_ = optimize.curve_fit(np.vectorize(gaussCustom), x1, y1, p0=guess1)
    params2, cov2, *_ = optimize.curve_fit(np.vectorize(gaussCustom), x2, y2, p0=guess2)
    a1, b1, amp1, mu1, sig1 = params1[0], params1[1], params1[2], params1[3], params1[4]
    a2, b2, amp2, mu2, sig2 = params2[0], params2[1], params2[2], params2[3], params2[4]
    fit_data1 = [gaussCustom(x1[i], a1, b1, amp1, mu1, sig1) for i in range(len(x1))]
    fit_data_linear1 = [a1 * x1[i] + b1 for i in range(len(x1))]
    fit_data2 = [gaussCustom(x2[i], a2, b2, amp2, mu2, sig2) for i in range(len(x2))]
    fit_data_linear2 = [a2 * x2[i] + b2 for i in range(len(x2))]
    print('Fit Parameters (Dataset A): ')
    for p in range(len(params1)):
        print(GAUSS_CUSTOM_PARAMETERS[p] + ': ', params1[p], ' +/- ', (cov1[p][p]) ** 0.5)
    print('Fit Parameters (Dataset B): ')
    for p in range(len(params2)):
        print(GAUSS_CUSTOM_PARAMETERS[p] + ': ', params2[p], ' +/- ', (cov2[p][p]) ** 0.5)
    plt.xlabel('Energy (keV)')
    plt.ylabel('Number of events')
    plt.plot(x1, y1, 'o', markersize=3, color='purple', label='Data S1')
    plt.plot(x1, fit_data1, '-', color='blue', label='Fit S1')
    plt.plot(x1, fit_data_linear1, '-', color='cyan', label='Base S1')
    plt.plot(x2, y2, 'o', marker='x', markersize=3, color='red', label='Data S2')
    plt.plot(x2, fit_data2, '--', color='orange', label='Fit S2')
    plt.plot(x2, fit_data_linear2, '--', color='yellow', label='Base S2')
    plt.legend()
    plt.show()


def plotCalcBkg(spectrum, bkg):
    spectrum.calcCountRate()
    bkg.calcCountRate()
    fig, ax1 = plt.subplots(1, constrained_layout=True)
    fig.suptitle(spectrum.name + ': count rate before and after the removal of background events')
    ax1.set(xlabel='Energy (keV)', ylabel='Count rate')
    ax1.set_xlim([0, 1500])
    ax1.step(spectrum.rebin_bins, spectrum.count_rate_bin_data, color='purple', label='137Cs')
    ax1.step(bkg.rebin_bins, bkg.count_rate_bin_data, color='green', label='Bkg')
    spectrum.subtractCountRateBkg(bkg)
    ax1.step(spectrum.rebin_bins, spectrum.count_rate_bin_data, color='black', label='137Cs-bkg')
    plt.legend()
    fig.canvas.manager.full_screen_toggle()
    plt.show()


def plotBinData(spectrum):
    fig, ax1 = plt.subplots(1)
    ax1.set_title(spectrum.path)
    ax1.step(spectrum.rebin_bins, spectrum.count_rate_bin_data, color='black', where='post')
    fig.canvas.manager.full_screen_toggle()
    plt.grid(True)
    plt.show()


def plotBinaryClassificationResults(method='sigma'):
    fig, axes = plt.subplots(3, constrained_layout=True)
    fig.set_size_inches(6, 12)
    for axis in axes:
        axis.set_ylabel('Accuracy')
        axis.grid(True)
        axis.set_ylim([0, 1.05])
    if method == 'sigma':
        plt.suptitle('"Sigma" method')
        axes[0].plot([25, 50, 100, 150, 200], [1, 1, 1, 1, 1], color='black', marker='o', markersize=3)
        axes[0].set_xlabel('Distance to the source, cm')
        axes[1].plot([1, 3, 5, 10, 30], [1, 1, 1, 1, 1], color='black', marker='o', markersize=3)
        axes[1].set_xlabel('Spectrum acquisition live time, s')
        axes[2].set_xlabel('Spectrum acquisition live time, s')
        axes[2].errorbar([1, 3, 5, 10, 30], [0.94, 0.98, 0.98, 0.98, 0.96],
                         yerr=[0.033585711, 0.01979899, 0.01979899, 0.01979899, 0.027712813], marker='o', color='black',
                         markersize=3, capsize=2)
    if method == 'mlrf':
        def erfunc(x, mFL, a, b, c):
            return mFL * scipy.special.erf((x - a) / (b * np.sqrt(2))) + c

        plt.suptitle('"MLRF" method')
        axes[0].errorbar([25, 50, 100, 150, 200], [1, 1, 0.1, 0.1, 0.1], yerr=[0, 0, 0.095, 0.095, 0.095], marker='o',
                         color='black', markersize=3, capsize=2)
        axes[0].set_xlabel('Distance to the source, cm')
        axes[1].errorbar([1, 3, 5, 10, 30], [1, 1, 1, 1, 1], yerr=[0, 0, 0, 0, 0], marker='o', color='black',
                         markersize=3, capsize=2)
        axes[1].set_xlabel('Spectrum acquisition live time, s')
        axes[2].set_xlabel('Spectrum acquisition live time, s')
        axes[2].errorbar([1, 3, 5, 10, 30], [0.94, 0.98, 0.98, 0.98, 0.98],
                         yerr=[0.033585711, 0.01979899, 0.01979899, 0.01979899, 0.01979899], marker='o', color='black',
                         markersize=3, capsize=2)
        x_data = np.linspace(0, 200, 200)
        axes[0].plot(x_data, erfunc(x_data, -0.45, 75, 10, 0.55), color='red')
    plt.show()


def mlShowLinfit(spectrum, sp_a, sp_b, bins_per_sect=settings.ml_bin_clf_bins_per_section):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    num_of_sections = int(len(spectrum.rebin_bins) / bins_per_sect)
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax1.step(spectrum.rebin_bins, spectrum.count_rate_bin_data, color='black', where='post')
    for section in range(num_of_sections):
        x = np.linspace(section * bins_per_sect, (section + 1) * bins_per_sect)
        ax1.vlines(section * bins_per_sect, ymin=-0.2,
                   ymax=0.2, color='r')
        ax1.plot(x, linear(x, sp_a[section], sp_b[section]), color='b')
    custom_lines = [Line2D([0], [0], color='black', lw=3),
                    Line2D([0], [0], color='red', lw=3),
                    Line2D([0], [0], color='blue', lw=3)]
    ax1.legend(custom_lines, ['Spectrum data', 'Section borders', 'Linear fit (by section)'])
    plt.ylabel('Count rate')
    plt.xlabel('Energy (keV)')
    plt.show()


def mlShowAverage(spectrum, sp_c, bins_per_sect=settings.ml_bin_clf_bins_per_section):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    num_of_sections = int(len(spectrum.rebin_bins) / bins_per_sect)
    fig, ax1 = plt.subplots(constrained_layout=True)
    fig.set_size_inches(12, 6, forward=True)
    ax1.step(spectrum.rebin_bins, spectrum.count_rate_bin_data, color='black', where='post')
    for section in range(num_of_sections - 1):
        x = np.linspace(section * bins_per_sect, (section + 1) * bins_per_sect)
        ax1.vlines(section * bins_per_sect, ymin=-0.2,
                   ymax=sp_c[section - 1], color='r')
        ax1.vlines(section * bins_per_sect, ymin=-0.2,
                   ymax=sp_c[section], color='r')
        ax1.plot([x[0], x[-1]], [sp_c[section], sp_c[section]], color='blue')
    custom_lines = [Line2D([0], [0], color='black', lw=3),
                    Line2D([0], [0], color='red', lw=3),
                    Line2D([0], [0], color='blue', lw=3)]
    ax1.legend(custom_lines, ['Spectrum data', 'Section borders', 'Average fit (by section)'])
    plt.ylabel('Count rate')
    plt.xlabel('Energy (keV)')
    plt.show()


def plotClassificationResults(spectrum, results, act_results, show_results=True, export=True, show=True, vis=None):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(12, 6, forward=True)
    plt.grid(True)
    plt.rcParams['font.family'] = 'monospace'
    fig.suptitle(spectrum.path, weight='bold')
    ax1.step(spectrum.rebin_bins, spectrum.count_rate_bin_data, color='black', where='post')
    plt.xlim(0, settings.kev_cap)
    plt.ylim(0, max(spectrum.count_rate_bin_data) * 1.2)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.subplots_adjust(right=0.98, left=0.05, top=0.88, bottom=0.18)
    plt.ylabel('Count rate')
    plt.xlabel('Energy (keV)')

    info = f'CHANNELS:  {spectrum.channel_qty}\n' \
           f'LIVET (s): {spectrum.live_time_int}\n' \
           f'REALT (s): {spectrum.real_time_int}\n' \
           f'DIST (cm): {int(spectrum.distance_from_src)}'

    ax1.text(0.87, 0.96,
             info,
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white'))

    ax2_ticks = []
    for key, val in spectrum.peak_data.items():
        for key in val.keys():
            if key > settings.clf_show_threshold:
                ax2_ticks.append(key)
    ax2 = ax1.secondary_xaxis('top')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_xticks(ax2_ticks, minor=False)

    isotope_lines = []
    for key, value in results.items():
        if value > settings.clf_display_threshold:
            for i in range(len(isodata.clf_isotopes[key].lines)):
                line = plt.vlines(isodata.clf_isotopes[key].lines[i],
                                  spectrum.count_rate_bin_data[isodata.clf_isotopes[key].lines[i]],
                                  max(spectrum.count_rate_bin_data) * 1.2,
                                  color=isodata.clf_isotopes[key].color,
                                  linewidth=2, alpha=value, zorder=0)
                if i == 0:
                    isotope_lines.append(line)

    for isotope_name, value in spectrum.peak_data.items():
        for line_kev, peak in value.items():
            plt.fill_between(spectrum.rebin_bins[peak.left_b:peak.right_b],
                             spectrum.count_rate_bin_data[peak.left_b:peak.right_b],
                             color=peak.isotope.color, step='post', alpha=results[isotope_name])
            plt.fill_between(spectrum.rebin_bins[peak.left_b:peak.right_b],
                             [peak.calc_bin_under_baseline(x) for x in range(peak.left_b, peak.right_b)],
                             color='black', alpha=0.3 * results[isotope_name], step='post',
                             edgecolor=peak.isotope.color)

    iso_legend_text = [f'{f"{isodata.clf_isotopes[key].name}:":<14}'
                       f'{f" {round(int(value * 100), 5)}%":<7} ~{round(act_results[key] / 1000):<3}'
                       f'{isUncertain(round(act_results[key] / 1000))} kBq'
                       for key, value in results.items()]
    legend = fig.legend(isotope_lines, iso_legend_text, loc='lower center', frameon=1, fancybox=False,
                        ncol=math.ceil(len(isotope_lines) / 2))
    frame = legend.get_frame()
    frame.set_edgecolor('black')

    if show_results:
        plt.ioff()
        plt.show()
    if export:
        img_path = f'{settings.images_path.joinpath(Path(spectrum.path).resolve().stem)}.png'
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        if show and vis is not None:
            vis.show_image(img_path)


def isUncertain(val):
    return '?' if val > isodata.cal_act_uncertainty_threshold_kBq else ''
