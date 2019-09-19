# Documentation

- [clean_db](#cleandb)

- [sdppg_features](#sdppgfeatures)
-----
# clean_db

#### clean_db.db_to_dataframe(filename)

  Load the json file obtained through create_db into a pandas dataframe

  Parameters
  ----
  filename: string; the full path to the json file, complete with extension

  Returns
  ----
  d: DataFrame; the dataframe containing the elements stored in the file

  Example
  ----
    >>>import pandas as pd
    >>>import json
    >>>import os
    >>>d = {'first': {'a': 1, 'b': 2}, 'second': {'a': 7, 'b': 14}}
    >>>with open('a.json', 'w') as file: json.dump(d, file)
    >>>f = db_to_dataframe('a.json')
    >>>fs
        Out:
        a   b
        0  1   2
        1  7  14
    >>>os.remove('a.json')
----

#### clean_db.clean_db(db, drop_columns=['rhythm'], drop_na_in=['weight', 'tpr', 'madRR', 'medianRR', 'opt_delay', 'afib', 'age', 'sex'], drop_zeroes_in=['weight', 'age'], quality_threshold=None, reset_index=True)

  Clean the 'cardio' dataframe by removing the 'bad' elements

  Parameters
  ----
  db: pandas DataFrame; the dataframe contaning the cardio data extracted through create_db

  drop_columns: list of strings; (optional; default: ['rhythm']); the column labels that must be dropped from the dataframe. If None, it is skipped
  
  drop_na_in: list of strings; (optional; default: 'weight', 'tpr', 'madRR', 'medianRR', 'opt_delay', 'afib', 'age', 'sex']); the subset of labels for which we must drop the row with NAN values. If None, it is skipped
  
  drop_zeroes_in: list of strings; (optional; default: ['weight', 'age']); the subset of labels for which we must drop the zero values because they bear no meaning. If None, it is skipped.
  
  quality_threshold: float; (optional; default: None); the quality threshold above which we drop the elements in the dataframe. BEWARE: if it is not None, the label 'quality' must be present!
  reset_index: bool; (default: True); if True reset the index so as to be able to call the elements by row

  Return
  ---
  newdb: pandas DataFrame; the cleaned dataframe

  Example
  ---
    >>>import numpy as np
    >>>import pandas as pd
    >>>d = {}
    >>>names = ['first', 'second', 'third', 'fourth']
    >>>v1, v2 = [0, 0, 56, 66], [20, np.nan, 40, 56]
    >>>for name, _, __ in zip(names, v1, v2):
    ...  d[name] = {'rhythm': np.nan, 'weight': _, 'age': __}
    >>>d = pd.DataFrame(d).T
    >>>clean_db(d)
        Out:
            age  weight
        0  40.0    56.0
        1  56.0    66.0
-----

# sdppg_features

#### sdppg_features.find_first_index_for_maximum_in_zero_crossing(sdppg, zero_crossing)

Find the first index in the zero_crossing array that corresponds to the
  leftmost end of a zero crossing interval for a local maximum.

  Parameters
  ----
  sdppg: 1D array-like of float; the sdppg signal

  zero_crossing: 1D array-like of float; the zero crossing found for sdppg through its second derivative

  Returns
  ----
  start: int; leftmost end of the interval containing a local maximum

  Example
  ----
    >>>import numpy as np
    >>>x = np.linspace(-2.5*np.pi, 2.5*np.pi, 2000)
    >>>f = np.sin(x)
    >>>d2f = np.gradient(np.gradient(f, x), x)
    >>>zero_crossing = np.where(d2f[0:-1]*d2f[1:] < 0.)[0]
    >>>s = find_first_index_for_maximum_in_zero_crossing(f, zero_crossing)
    >>>i1, i2 = zero_crossing[s], zero_crossing[s+1]
    >>>np.max(f[i1:i2])  # expected 1: max(sin(x))
        0.9999987650650776
    >>>x[np.argmax(f[i1:i2])+i1]  # must be shifted by i1; expected -3*np.pi/2 (~ -4.712)
        -4.710817398266836

----

#### sdppg_features.find_next_a(sdppg, zero_crossing, start)
"""
  Find the next a peak in the SDPPG signal.

  Parameters
  ----
  sdppg: 1D array-like of float; the (normalised) sdppg signal
  zero_crossing: 1D array-like of float; the zero crossing found from the fourth derivative of the PPG
  start: int, starting position from which the zero_crossing array will be read

  Returns
  ----
  a: float; the feature
  k: int; the first position in the zero-crossing array after a

  Notes: if there is no next a, a is set to np.nan and k to len(zero_crossing)
  For further information:
      https://ieeexplore.ieee.org/document/5412099

  Example
  ----
    >>># simulating a PPG signal as a sum of sinusoidal waves with frequences 1, 2, and 3 Hz
    >>>t = np.linspace(-0.3, 1., 600)
    >>>v = np.asarray([1., 2., 3.])
    >>>s = 0
    >>>for f in v:
    ...  s += f**(-2)*np.sin(2*np.pi*f*t)
    >>>sdppg = np.gradient(np.gradient(s, t), t)
    >>>fdppg = np.gradient(np.gradient(sdppg, t), t)
    >>>sdppg = np.flip(sdppg)
    >>>fdppg = np.flip(fdppg)
    >>>zero_crossing = np.where(fdppg[0:-1]*fdppg[1:] < 0.)[0]
    >>>start = find_first_index_for_maximum_in_zero_crossing(sdppg, zero_crossing)
    >>>find_next_a(sdppg, zero_crossing, start)  # (a, k)
        Out:
        (98.64934136508828, 1)

----

#### sdppg_features.sdppg_agi(a, b, c, d, e)

Compute the ageing index (AGI) from the features (a, b, c, d, e) extracted
  from the SDPPG approach. The formula is (b-c-d-e)/a. The feature can be obtained through
  features_from_sdppg.

  Parameters
  ----
  a: float; array-like of float; feature 'a' extracted from the sdppg approach.
  
  b: float; array-like of float; feature 'b' extracted from the sdppg approach.
  
  c: float; array-like of float; feature 'c' extracted from the sdppg approach.
  
  d: float; array-like of float; feature 'd' extracted from the sdppg approach.
  
  e: float; array-like of float; feature 'e' extracted from the sdppg approach.

  Returns
  ----
  AGI: numpy array of floats; the ageing index computed for each element.

  For further reading:
      https://www.hindawi.com/journals/tswj/2013/169035/abs/
      https://ieeexplore.ieee.org/document/5412099

  Example
  ----
    >>>a = [1, .8, .3]
    >>>b = [-.9, -.7, -.2]
    >>>c = [.1, .1, .1]
    >>>d = [-.1, -.1, -.1]
    >>>e = c
    >>>sdppg_agi(a, b, c, d, e)
    Out:
        [-1., -1., -1.]

----

#### sdppg_features.features_from_sdppg(t, signal, normalise=True, flip=True, spline=True, f=100)

Find the a, b, c, d, e parameters (features, "waves") for a PPG signal
  through its normalised second derivative.

  Parameters
  ----
  t: 1D array-like of float; the time points at which the signal has been measured
  
  signal: 1D array-like of float; the measured PPG signal
  
  normalise: bool(default=True); if True, normalise SDPPG and its second derivative
  
  flip: bool (default=True); should the signal be flipped?
  
  spline: bool (default=True); use the cubic spline interpolation of signal instead of signal. BEWARE of splines: they can change the maxima and minima value and position!
  
  f: int (default=100); factor used to compute the new number of points if spline==True : the length of the splined signal will be len(t)*f

  Returns
  ----
  sdppg: the sdppg computed from the PPG signal provided. It is the result of the spline if spline=True
  
  features: dictionary containing 10 numpy array: a, b, c, d, e (each of them is a np array of the respective "wave" fount in the SDPPG), the AGI index (AGI) computed for each element of the array, and the time differences between the "waves" a and b, b and c, c and d, and d and e (t_ab, t_bc, t_cd, t_de).

  NOTES
  ---
  t and signal must have the same length; t must be monotonic and positive

  a is the initial positive wave

  b is early negative wave (we suppose b < 0)

  c is the re-upsloping wave

  d is the re-downloping wave

  e is the diastolic positive wave

  For further information, please read:
      https://www.hindawi.com/journals/tswj/2013/169035/abs/
      https://ieeexplore.ieee.org/document/5412099

  Example
  ----

    >>># simulating a PPG signal as a sum of sinusoidal waves with frequences 1, 2, and 3 Hz
    >>>t = np.linspace(-0.3, 1, 600)
    >>>w = 2*np.pi
    >>>v = np.asarray([1., 2., 3.])
    >>>s = 0
    >>>for f in v:
    ...  s += f**(-2)*np.sin(w*f*t)
    >>>f, a = features_from_sdppg(t, s, normalise=False, spline=False)
    >>>a
        Out:
        {'a': array([98.64934137]),
        'b': array([-9.54744371]),
        'c': array([25.06845097]),
        'd': array([-25.07132544]),
        'e': array([9.54438823]),
        'AGI': array([-0.19350314]),
        't_ab': array([0.18230384]),
        't_bc': array([0.13238731]),
        't_cd': array([0.15843072]),
        't_de': array([0.13238731])}  

----
