
# -*- coding: utf-8 -*-

# cardio stuff
import clean_db
import create_db
import pre_process as pr

# standard libraries
import numpy as np
from scipy.optimize import curve_fit
from operator import itemgetter



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
  for i in np.array([.5, 1., 1.5, 2., 3.]):
    mt, mov_avg = pr.m_avg(time, sign, int(srate*i))

    # use len_filler to make mov_avg the same size as sign
    len_filler = np.zeros((len(sign)-len(mov_avg))//2) + np.mean(sign)
    mov_avg = np.insert(mov_avg, 0, len_filler)
    mov_avg = np.append(mov_avg, len_filler)

    peaklist, sign_peak = create_db.detect_peaks(sign, mov_avg)

    # keeping only peaks detected with all 5 different windows
    peaks = np.intersect1d(peaks, peaklist)

  # first element can't be a correct local extrema, it has no points before
  if(peaks[0] == 0):
    peaks = np.delete(peaks, 0)

  # last element can't be a correct local extrema, it has no points after
  if(peaks[-1] == len(sign)-1):
    peaks = np.delete(peaks, -1)

  # peak checking: rejecting lower peaks where RR distance is too small
  final_peaks = []  # definitive peak positions container
  last_peak = -1  # parameter to avoid undesired peaks still in final_peaks
  for p in peaks:
    if p <= last_peak:
      continue

    evaluated_peaks = [g for g in peaks if p <= g <= srate*.5+p]
    last_peak = max(evaluated_peaks)
    final_peaks.append(evaluated_peaks[np.argmin([sign[x] for x in evaluated_peaks])])

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

  correct_x_of_maxima: list; list containing the x component of the maxima of
  each single peak
  """
  x_of_minima = find_x_of_minima(time, signal)

  parameter_list = []
  total_beat_duration_list = []
  total_beat_height_gap_list = []
  correct_x_of_maxima = []

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
    X = np.asarray(time[x_of_minima[i]:x_of_minima[i+1]+1])
    Y = np.asarray(signal[x_of_minima[i]:x_of_minima[i+1]+1])
    Y = Y - min(Y)

    lower_bounds = np.array([0, min(X), 0., -max(Y), min(X), 0.])
    upper_bounds = np.array([max(Y), max(X), max(X)-min(X), max(Y), max(X),
                             max(X)-min(X)])

    # to initialize the 2 gaussians with different parameters we use pre_gaus
    # on 2 different portions of the peak ([:split] and [-split:])
    # we tried to vary split from 25% up to 75% of the peak lenght and it
    # did not affected the output values of the fit for every test peaks used
    # so we supposed it fixed to half of the lenght just for simplicity
    split = int(len(X)*3/4)

    try:

      # NOTE the double "(" after np.concatenate is needed to work properly,
      # since conatenate must work on a tuple in our case.
      parameters, errors = curve_fit(gaus, X, Y,
                                     p0=np.concatenate((pre_gaus(X[:split],
                                                                Y[:split]),
                                                       pre_gaus(X[-split:],
                                                                Y[-split:]))),
                                     bounds=(lower_bounds, upper_bounds))

      parameter_list.append(parameters)
      total_beat_duration_list.append(max(X)-min(X))
      total_beat_height_gap_list.append(max(Y)-min(Y))
      correct_x_of_maxima.append(X[np.argmax(Y)])

    # skip problematic peaks
    except RuntimeError:
      continue
    except TypeError:
      continue

  return parameter_list, total_beat_duration_list, total_beat_height_gap_list, correct_x_of_maxima
