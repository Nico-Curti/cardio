
# -*- coding: utf-8 -*-

# cardio stuff
import create_db
import pre_process as pr

# standard libraries
import numpy as np
from scipy.optimize import curve_fit


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

  Example
  ----
  >>>import numpy as np
  >>>
  >>>x = np.linspace(0., 10*np.pi, 10000)
  >>>y = np.sin(x)
  >>>x_of_minima = find_x_of_minima(x, y)
  >>>
  >>>x_of_minima
    Out:
    array([1500, 3500, 5499, 7499])
  >>>x[x_of_minima]
    Out:
    array([ 4.71286027, 10.99667395, 17.27734574, 23.56115943])
  >>>y[x_of_minima]
    Out:
    array([-0.99999989, -0.9999994 , -0.999999  , -0.99999969])
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

  # this check will be repeated after possible elimination of first and last peak
  if(len(peaks) < 2):
    raise ValueError(
      "found {} minima BEFORE their validity check. We need at least 2 "
      "to identify an isolated peak and perform fit. Maybe you are not using "
      "a long enough portion of signal. If you want to analyze a single "
      "peak make sure you have at least 1 detectable local minima before and "
      "1 after it".format(len(peaks)))

  # first element can't be a correct local extrema, it has no points before
  if(peaks[0] == 0):
    peaks = np.delete(peaks, 0)

  # last element can't be a correct local extrema, it has no points after
  if(peaks[-1] == len(sign)-1):
    peaks = np.delete(peaks, -1)

  # repeating check after
  if(len(peaks) < 2):
    raise ValueError(
      "found {} minima AFTER their validity check, but we need at least 2 "
      "to identify an isolated peak and perform fit. Please note that the "
      "first/last point of the signal can not be a local minimum since "
      "it does not have a point before/after. Maybe you are not using "
      "a long enough portion of signal. If you want to analyze a single "
      "peak make sure you have at least 1 detectable local minima before and "
      "1 after it".format(len(peaks)))


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
  diastolic peaks by performing a double gaussian fit.
  The double gaussian fit is performed between two local minima,
  so it needs a local minimum before the systolic peak and one after the
  diastolic peak to work properly.

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

  Example
  ----
  >>>import numpy as np
  >>>
  >>>state = np.random.RandomState(seed = 42)
  >>>
  >>>mean1, var1 = 0. , .5
  >>>mean2, var2 = 2. , 1.
  >>>
  >>>gauss1 = state.normal(mean1, var1, 1000000)
  >>>gauss2 = state.normal(mean2, var2, 1000000)
  >>>
  >>># will be used also as x-axis
  >>>binning = np.linspace(-4., 7., 100)
  >>>
  >>># summing the 2 gaussians, will be used as y-axis
  >>>double_gaussian = np.histogram(gauss1, bins=binning)[0] + np.histogram(gauss2, bins=binning)[0]
  >>>
  >>># adding two spikes at either side of double_gaussian in order to obtain
  >>># two local minima (one before and one after the signal) because they
  >>># are needed for features_from_dictrotic_notch to identify the peak properly
  >>>double_gaussian[1] = max(double_gaussian)
  >>>double_gaussian[-1] = max(double_gaussian)
  >>>
  >>>
  >>>features = features_from_dicrotic_notch(binning[:-1], double_gaussian)
  >>>
  >>>print("mean1:", mean1, "    predicted mean1:", features[0][0][1])
  >>>print("var1:", var1, "    predicted var1:", features[0][0][2])
  >>>print("mean2:", mean2, "    predicted mean1:", features[0][0][4])
  >>>print("var2:", var2, "    predicted mean1:", features[0][0][5])
    Out:
    mean1: 0.0     predicted mean1: -0.055731720038204834
    var1: 0.5     predicted var1: 0.5020164668217136
    mean2: 2.0     predicted mean1: 1.9465910153756594
    var2: 1.0     predicted mean1: 0.9993471020376136
  """
  x_of_minima = find_x_of_minima(time, signal)

  parameter_list = []
  total_beat_duration_list = []
  total_beat_height_gap_list = []

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

    # skip problematic peaks
    except RuntimeError:
      continue
    except TypeError:
      continue

  return parameter_list, total_beat_duration_list, total_beat_height_gap_list
