# **XGD-MLC**
*A Python3 ML classifier of radioactive gamma-sources, designed to work with the energy spectra acquired by Xenon Gamma-Detector of NRNU MEPhI. Developed for the master's thesis by Ivan Davydov (2023).*

## What does it do?
XGD-MLC software uses scikit-learn ML-algorithms, enabling it to predict:
* Are radioactive source signatures (energy peaks) present in a given gamma-spectrum;
* If there are peaks present, what gamma emmitters (isotopes) caused them;
* Detected isotopes' approximate decay rate (activity).

## What does all of this mean, briefly?
**Gamma-rays (gamma-quanta)** are basically deadly, very high-energy *photons* (yes, the ones that your eyes absorb right now to make you able to read all of this). An enourmous variety of different devices exist that are capable of detecting such hard-hitting particles, all created for their own scientific goal. One of such detectors is a **Xenon Gamma-Spectrometer**, developed by department №7 of Moscow's **NRNU MEPhI** (below I'll leave a link to my uni's paper thouroughly describing it, if you are interested in how this piece of machinery operates).

In its "raw" state, gamma-spectrum is just a sequence of X and Y values, binary-encoded to a *.sps* file with additional useful data regarding the specifics of how it was recorded (distance from the detector's body to the radioactive source, live/actual spectrum recording time, etc -- *more details below*).
It is basically a histogram, where X-values are proportional to energy and correspond to ADC's channels (it transforms analog signal from the detector to a digital form), and Y-values are quantities of gamma-ray registrations in a given ADC channel. To give this histogram physical meaning, it first has to be calibrated. This process streches and displaces the histogram in such a way that allows to present X-values as keV's. This action of resizing the X-axis makes bins hard to work with -- so they are standartized (rebinned: [0, 1, 2, 3...]) and divided by live spectra acquisition time, transforming the Y-values from counts to count rates.

## Why would this be useful to anyone?
A valid energy spectra of an object or environment can be used to make conclusions about the presence of gamma-emitting nuclei, which makes their analisys useful for:

* radioecology investigations of irradiated territories;
* radioactive item control in ports, customs, checkpoints, military sites;
* gamma-emitting sample analysis --

and many more specialized or civilian applications, many of which would benefit if the whole process of analyzing the spectra was automatized (and didn't require a nuclear physicist on-site at all times to help) or, at least, it would make analysis of massive amounts of spectra much faster and simpler. Implementing Machine Learning helps to achieve this.

## Machine learning?

Loads of painfully complicated math (which I had a fortune of not developing, thank god scikit-learn exists), which enables a machine to do a given task after being taught to complete sets of similar tasks. In our case, we supply the algorithm with a large amount of various spectra sets (the ones we want it to distinguish from each other) -- spectra without any sources nearby (background spectra), spectra containing Cs137 signatures, Am241 signatures, Co60 -- and so on. These .sps filesets should be varied (by spectra acquirement time, distance to source) and balanced (roughly similar quantities of each kind) for the best classification accuracy. There are a few models, all working very different "under the hood". After selecting one, we need to feed it with portions of data (features), which best resemble our spectra, so the algorithm won't be overwhelmed. During the development I came up with 2 types of features - LINFIT and AVERAGE. 

* LINFIT works by splitting the spectrum into a custom (m) number of segments, and then fitting a linear function for every one. Resulting features - linear parameters of every segment [a1, a2, ..., am, b1, b2, ..., bm] are placed in a pandas dataframe to be used in model creation.


* AVERAGE is more simple (and effective) - it similarly slices a given spectrum into m slices, after which the average count rate for each one is computed. The values are then placed into a similar dataframe [c1, c2, ..., cm].

And then you can scale them, for example, by applying tan() to every one. In some cases, it can make the classification results more accurate. 

At first, the whole dataset gets binary-classified (has gamma-signature / doesn't have gamma-signature). Then comes the multi-label classification step: spectra with detected energy peaks are processed further. Multiple nonexclusive categories may be assigned to each instance -- for example, a spectrum may contain both Cs137 and Co60 energy peaks. It also provides the user with a measure of certainty of its prediction regarding the presence of detected isotopes' signatures in the spectrum. 

Both binary and multilabel steps require different datasets and models, so they are stored in program's filesystem separately.


## 😐...

Indeed! To sum it up, these are the general steps the program does to give user a classification result:

1. Loads .sps fileset (to classify) from its designated directory (sps/test by default), placing the spectra data into instances of *Spectrum* class;
2. Processes each one (calibrate, rebin, calculate count rate and required parameters, remove redundant data and corrupted spectra);
3. Proceeds to binary classification;
4. Checks if there is already a previously created binary ML-model file available in /models/binary (with the same configuration as currently set by user). If yes, proceed to step 9;
5. No binary model found - checks if there is already a previously created binary dataframe file available in /dataframes/binary (with the same configuration as currently set by user). If yes, skip to step 8;
6. Loads .sps filesets (to learn) from their designated directories (sps/srcs, sps/bkgs by default), placing the spectra data into instances of *Spectrum* class;
7. Finds features (selected by user) in all datasets, creates a new dataframe;
8. Creates a user-selected binary model, evaluates its preliminary performance;
9. Performs binary classification. Exports results to report file (default path /reports/bin_clf_report.txt), calculates the error matrix and prints it to CLI;
10. Proceeds to multi-label classification (if configured - if not, stop);
11. Checks if there is already a previously created multi-label ML-model file available in /models/multilabel (with the same configuration as currently set by user). If yes, proceed to step 16;
12. No multi-label model found - checks if there is already a previously created multilabel dataframe file available in /dataframes/multilabel (with the same configuration as currently set by user). If yes, skip to step 15;
13. Loads .sps filesets (to learn) from their designated directories (sps/srcs, sps/bkgs by default), placing the spectra data into instances of *Spectrum* class;
14. Finds features (selected by user) in all datasets, creates a new dataframe;
15. Creates a user-selected multi-label model, evaluates its preliminary performance;
16. Performs multi-label classification. Shows the gamma-spectrum classification result graph (detected peaks, probabilities, gamma-source activities - exports it to /images, if configured). Exports results to report file (default path /reports/clf_report.txt), calculates metrics (EMR, Accuracy, Precision) and prints them to CLI;
17. Proceeds to activity prediction (if configured). Calculates peak data, creating *Peak*-class objects;
18. Exits.



## Whatever. How do I try it out?

Clone the software, place it wherever you like. Now, you will need a dataset. Here's the one I acquired using the XGD (several thousand spectra -- it took... hours and hours) and used during development and testing:

With this dataset, you will be able to work with Na22, Co60, Cs137, Eu152 and Am241 (the latter two are the tricky ones, you will see why). Extract the files from the archive into the program's root directory. After that, you can either configure various parameters via config/settings.py file, or straight up start up the program via the terminal (or your preferred IDE) and use supported modifiers there to change classification parameters. Here's what CLI can process:

# TABLE

Here's some examples:

# EXAMPLES

## It doesn't work / something's broken / your code sucks!
If that's the case, please, notify me. I plan to keep working on this for some time even after I defend my master's thesis. I will try to fix the issue. Also tell me if you find any inefficiencies / horrible code practices (I'm here to study). Feel free to implement and improve my code in your own projects, if you found it useful (I'd be grateful if you at least mention me, though!)
