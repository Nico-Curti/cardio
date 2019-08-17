
# -*- coding: utf-8 -*-

# cardio stuff
import clean_db
import create_db
import pre_process as pr

# standard libraries
import numpy as np
from scipy.optimize import curve_fit
from operator import itemgetter

# %%

# Load json db and convert it to dataframe
f = "cardio.json"

df = clean_db.db_to_dataframe(f)

# %%

# Clean file for further use
db = clean_db.clean_db(df,
                       drop_columns=['rhythm', 'city', 'country',
                                     'filename', 'filenumber'],
                       drop_na_in=['weight', 'tpr', 'madRR', 'medianRR',
                                   'opt_delay', 'afib', 'age', 'sex', 'smoke',
                                   'afib', 'bmi', 'lifestyle'],
                       drop_zeroes_in=['weight', 'age', 'length'],
                       quality_threshold=None, reset_index=True)

# CONVERTING STRING LABELS TO ARBITRARY NUMERICAL VALUES
db = db.replace('F', -1).replace('M', 1)
db = db.replace('C1', 1).replace('C0', 0).replace('C3', 3).replace('C2', 2)
db = db.replace('Active', 3
                ).replace('Actief', 3
                ).replace('Gemiddeld', 2
                ).replace('Moderate', 2
                ).replace('Sedentary', 1
                ).replace('Weinig', 1)

# %%

# TODO insert this function in create_db or define it in create_db


def find_x_of_minima(time, signal):
  """
  find index position of local minima whose amplitude is under a certain
  moving threshold

  Parameters
  ----
  time: numerical 1-D array-like; basically the x axis of the curve whose
  minima will be found

  signal: numerical 1-D array-like; basically the y axis of the curve whose
  minima will be found

  Return
  ----
  final_peaks: list; the list containing the index positions of signal minima
  """

  sign = -1*np.asarray(signal)  # -1* is used to find minima instead of maxima

  # using time to extrapolate sampling rate
  srate = len(time)/(max(time)-min(time))
  peaks = np.arange(len(sign))  # initializing peaks index

  # different widths used for moving window, the unit is referred to 1 srate
  for i in np.array([.5, 1., 1.5, 2.]):
    mt, mov_avg = pr.m_avg(time, sign, int(srate*i))

    # use len_filler to make mov_avg the same size as sign
    len_filler = np.zeros((len(sign)-len(mov_avg))//2) + np.mean(sign)
    mov_avg = np.insert(mov_avg, 0, len_filler)
    mov_avg = np.append(mov_avg, len_filler)

    peaklist, sign_peak = create_db.detect_peaks(sign, mov_avg)

    # keeping only peaks detected with all 4 different windows
    peaks = np.intersect1d(peaks, peaklist)

  # peak checking: rejecting lower peaks where RR distance is too small
  final_peaks = []  # definitive peak positions container
  last_peak = -1  # parameter to avoid undesired peaks still in final_peaks
  for p in peaks:
    if p <= last_peak:
      continue

    evaluated_peaks = [g for g in peaks if p <= g <= srate*.5+p]
    last_peak = max(evaluated_peaks)
    final_peaks.append(evaluated_peaks[np.argmax([sign[x] for x in evaluated_peaks])])

  final_peaks = np.unique(final_peaks)  # to avoid repetitions

  return final_peaks


def features_from_dicrotic_notch(time, signal):
  """
  function used to extract features from differences between systolic and
  diastolic peaks by performing a double gaussian fit

  Parameters
  ----
  time: numerical 1-D array-like; basically the x axis of the curve to fit

  signal: numerical 1-D array-like; basically the y axis of the curve to fit

  Return
  ----
  parameter_list: list; list containing 6 parameters for each peak: maximum
  height, mean and standard deviation of the first gaussian and same for the
  second gaussian

  total_beat_duration_list: list; list containing the range on x axis covered
  by each single peak

  total_beat_height_list: list; list containing the range on y axis covered
  by each single peak
  """
  x_of_minima = find_x_of_minima(time, signal)

  parameter_list = []
  total_beat_duration_list = []
  total_beat_height_list = []

  # function to initialize the fitting parameters (to avoid wrong convergence)
  def pre_gaus(x, y):
    x = np.asarray(x)
    y = np.asarray(y)-min(y)

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    return [max(y)+min(y), mean, sigma]

  # function used in the fit
  def gaus(x, a1, m1, s1, a2, m2, s2):
    x = np.asarray(x)
    return a1*np.exp(-(x-m1)**2/(2*s1**2))+a2*np.exp(-(x-m2)**2/(2*s2**2))

  for i in range(len(x_of_minima)-1):
    # separate the peak into its x and y component
    X = time[x_of_minima[i]:x_of_minima[i+1]+1]
    Y = signal[x_of_minima[i]:x_of_minima[i+1]+1]

    # to initialize the 2 gaussians with different parameters we use pre_gaus
    # on 2 different portions of the peak ([:split] and [-split:])
    # we tried to vary split from 25% up to 75% of the peak lenght and it
    # did not affected the output values of the fit for every test peaks used
    # so we supposed it fixed to half of the lenght just for simplicity
    split = int(len(X)/2)

    try:

      # NOTE the double "(" after np.concatenate is needed to work properly,
      # since conatenate must work on a tuple in our case.
      parameters, errors = curve_fit(gaus, X, Y,
                                     p0=np.concatenate((pre_gaus(X[:split],
                                                                Y[:split]),
                                                       pre_gaus(X[-split:],
                                                                Y[-split:]))))

      parameter_list.append(parameters)
      total_beat_duration_list.append(max(X)-min(X))
      total_beat_height_list.append(max(X)-min(Y))

    # skip problematic peaks
    except RuntimeError:
      continue
    except TypeError:
      continue

  return parameter_list, total_beat_duration_list, total_beat_height_list


# %%

# =============================================================================
# EXTRACT FEATURES
# =============================================================================

notch_features = []
durations = []
heights = []

for guy in range(len(db.age)):
  print(guy)
  P, D, H = features_from_dicrotic_notch(db.time[guy], db.signal[guy])

  notch_features.append(P)
  durations.append(D)
  heights.append(H)
print("ended")
# %%

# =============================================================================
# COMPUTE FURTHER FEATURES
# =============================================================================

# to shorten the below variable names we used:
# n_ = normalized feature (it has been divided by total duration or height)
# dist_ = computed on the difference between peak positions
# diff_ = computed on the difference between peak heights
# mea = mean
# std = standard deviation
# med = median
# mad = median absolute deviation
# _all = indicates that every succesfully fitted peak has been used

# when "_all" is not present only the good ones have been used


dist_mea = []
dist_std = []
dist_med = []
dist_mad = []

diff_mea = []
diff_std = []
diff_med = []
diff_mad = []

n_dist_mea = []
n_dist_std = []
n_dist_med = []
n_dist_mad = []

n_diff_mea = []
n_diff_std = []
n_diff_med = []
n_diff_mad = []



dist_mea_all = []
dist_std_all = []
dist_med_all = []
dist_mad_all = []

diff_mea_all = []
diff_std_all = []
diff_med_all = []
diff_mad_all = []

n_dist_mea_all = []
n_dist_std_all = []
n_dist_med_all = []
n_dist_mad_all = []

n_diff_mea_all = []
n_diff_std_all = []
n_diff_med_all = []
n_diff_mad_all = []



for nf, d, h in zip(notch_features, durations, heights):

  a1 = np.asarray(list(map(itemgetter(0), nf)))
  m1 = np.asarray(list(map(itemgetter(1), nf)))
  a2 = np.asarray(list(map(itemgetter(3), nf)))
  m2 = np.asarray(list(map(itemgetter(4), nf)))


  peak_distances = m2-m1
  peak_differences = a2-a1

  normalized_peak_distances = peak_distances/d
  normalized_peak_differences = peak_differences/h

  # discard peaks where normalization isn't in [0;1]
  good_ones = np.logical_and(np.logical_and(normalized_peak_distances>0,
                                            normalized_peak_distances<1),
                             np.logical_and(normalized_peak_differences>0,
                                            normalized_peak_differences<1))



  dist_mea_all.append(np.mean(peak_distances))
  dist_std_all.append(np.std(peak_distances))
  dist_med_all.append(np.median(peak_distances))
  dist_mad_all.append(np.median(np.abs(peak_distances-
                                       np.median(peak_distances)
                                       )))

  diff_mea_all.append(np.mean(peak_differences))
  diff_std_all.append(np.std(peak_differences))
  diff_med_all.append(np.median(peak_differences))
  diff_mad_all.append(np.median(np.abs(peak_differences-
                                       np.median(peak_differences)
                                       )))

  n_dist_mea_all.append(np.mean(normalized_peak_distances))
  n_dist_std_all.append(np.std(normalized_peak_distances))
  n_dist_med_all.append(np.median(normalized_peak_distances))
  n_dist_mad_all.append(np.median(np.abs(normalized_peak_distances-
                                         np.median(normalized_peak_distances)
                                         )))

  n_diff_mea_all.append(np.mean(normalized_peak_differences))
  n_diff_std_all.append(np.std(normalized_peak_differences))
  n_diff_med_all.append(np.median(normalized_peak_differences))
  n_diff_mad_all.append(np.median(np.abs(normalized_peak_differences-
                                         np.median(normalized_peak_differences)
                                         )))



  peak_distances = peak_distances[good_ones]
  peak_differences = peak_differences[good_ones]
  normalized_peak_distances = normalized_peak_distances[good_ones]
  normalized_peak_differences = normalized_peak_differences[good_ones]

  dist_mea.append(np.mean(peak_distances))
  dist_std.append(np.std(peak_distances))
  dist_med.append(np.median(peak_distances))
  dist_mad.append(np.median(np.abs(peak_distances-
                                   np.median(peak_distances)
                                   )))

  diff_mea.append(np.mean(peak_differences))
  diff_std.append(np.std(peak_differences))
  diff_med.append(np.median(peak_differences))
  diff_mad.append(np.median(np.abs(peak_differences-
                                   np.median(peak_differences)
                                   )))

  n_dist_mea.append(np.mean(normalized_peak_distances))
  n_dist_std.append(np.std(normalized_peak_distances))
  n_dist_med.append(np.median(normalized_peak_distances))
  n_dist_mad.append(np.median(np.abs(normalized_peak_distances-
                                     np.median(normalized_peak_distances)
                                     )))

  n_diff_mea.append(np.mean(normalized_peak_differences))
  n_diff_std.append(np.std(normalized_peak_differences))
  n_diff_med.append(np.median(normalized_peak_differences))
  n_diff_mad.append(np.median(np.abs(normalized_peak_differences-
                                     np.median(normalized_peak_differences)
                                     )))


# %%

# =============================================================================
# add new features
# =============================================================================


db['dist_mea'] = dist_mea
db['dist_std'] = dist_std
db['dist_med'] = dist_med
db['dist_mad'] = dist_mad

db['diff_mea'] = diff_mea
db['diff_std'] = diff_std
db['diff_med'] = diff_med
db['diff_mad'] = diff_mad

db['n_dist_mea'] = n_dist_mea
db['n_dist_std'] = n_dist_std
db['n_dist_med'] = n_dist_med
db['n_dist_mad'] = n_dist_mad

db['n_diff_mea'] = n_diff_mea
db['n_diff_std'] = n_diff_std
db['n_diff_med'] = n_diff_med
db['n_diff_mad'] = n_diff_mad


db['dist_mea_all'] = dist_mea_all
db['dist_std_all'] = dist_std_all
db['dist_med_all'] = dist_med_all
db['dist_mad_all'] = dist_mad_all

db['diff_mea_all'] = diff_mea_all
db['diff_std_all'] = diff_std_all
db['diff_med_all'] = diff_med_all
db['diff_mad_all'] = diff_mad_all

db['n_dist_mea_all'] = n_dist_mea_all
db['n_dist_std_all'] = n_dist_std_all
db['n_dist_med_all'] = n_dist_med_all
db['n_dist_mad_all'] = n_dist_mad_all

db['n_diff_mea_all'] = n_diff_mea_all
db['n_diff_std_all'] = n_diff_std_all
db['n_diff_med_all'] = n_diff_med_all
db['n_diff_mad_all'] = n_diff_mad_all


# %%

# =============================================================================
# save new db
# =============================================================================

db.T.to_json('./db_after_notch_features.json')
