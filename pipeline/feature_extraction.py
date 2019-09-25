
# -*- coding: utf-8 -*-

# standard libraries
import numpy as np
from operator import itemgetter
import sys, argparse

# cardio stuff
sys.path.append("../cardio")
import clean_db, sdppg_features
import double_gaussian_features as dgf
sys.path.append("../pipeline")

if __name__ == '__main__':
  description = "Add features to cardio database through SDPPG, double gaussian features, ..."
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("-in", required=True, dest="input_json", action="store",
                      help="Input json file", default='')
  parser.add_argument("-out", required=False, dest="output_json", action="store",
                      help="Output json file. Default = cadio_final.json",
                      default="cardio_final.json")
  if len(sys.argv) <= 1:
    parser.print_help()
    sys.exit(1)
  else:
    args = parser.parse_args()

  # get parameters
  input_json = args.input_json
  output_json = args.output_json

  # load dataframe
  print("Loading database...")
  df = clean_db.db_to_dataframe(input_json)

  print("Extracting features from sdppg...")
  # get sdppg features
  features = []

  for _, __ in zip(df.time, df.signal):
    sdppg, ff = sdppg_features.features_from_sdppg(_, __, spline=False, normalise=False)
    features.append(ff)

  # add features to dataframe
  keys = list(features[0].keys())

  for key in keys:
    p = []
    for _ in features:
      p.append(np.median(_[key]))
    df[key] = p

  print("Extracting addictional features from the SDPPG features...")
  # compute temporal intervals
  df['t_ac'] = np.asarray([np.median(feature['t_ab'] + feature['t_bc']) for feature in features])
  df['t_ad'] = np.asarray([np.median(feature['t_ab'] + feature['t_bc'] + feature['t_cd']) for feature in features])
  df['t_ae'] = np.asarray([np.median(feature['t_ab'] + feature['t_bc'] + feature['t_cd'] + feature['t_de']) for feature in features])
  df['t_bd'] = np.asarray([np.median(feature['t_bc'] + feature['t_cd']) for feature in features])
  df['t_be'] = np.asarray([np.median(feature['t_bc'] + feature['t_cd'] + feature['t_de']) for feature in features])
  df['t_ce'] = np.asarray([np.median(feature['t_cd'] + feature['t_de']) for feature in features])
  # b, c, d, e features augmentation by dividing by a
  v = ['b', 'c', 'd', 'e']
  k = ['b/a', 'c/a', 'd/a', 'e/a']
  for _, __ in zip(k, v):
    df[_] = df[__] / df.a

  df['b-a'] = df['b'] - df['a']
  df['(b-e)/a'] = df['b/a'] - df['e/a']
  df['(c+d-b)/a'] = df['c/a'] + df['d/a'] - df['b/a']
  df['b - (d/a)'] = df['b'] - df['d/a']

  # Now compute the slope combinations for the waves a, b, c, d, e
  k = ['ab_slope', 'ac_slope', 'ad_slope', 'ae_slope',
       'bc_slope', 'bd_slope', 'be_slope',
       'cd_slope', 'ce_slope',
       'de_slope']
  v = [['a', 'b', 't_ab'], ['a', 'c', 't_ac'], ['a', 'd', 't_ad'], ['a', 'e', 't_ae'],
       ['b', 'c', 't_bc'], ['b', 'd', 't_bd'], ['b', 'e', 't_be'],
       ['c', 'd', 't_cd'], ['c', 'e', 't_ce'],
       ['d', 'e', 't_de']]
  for _, __ in zip(k, v):
    v1, v2, v3 = __
    y0, y1, dt = df[v1], df[v2], df[v3]
    df[_] = (y1 - y0)/dt

  print("Computing BMI...")
  # length is in cm
  df['bmi'] = df['weight'].astype(float)/((df['length'].astype(float))**2/1E4)


  print("Gaussian fitting...")
  # =============================================================================
  # EXTRACT FEATURES
  # =============================================================================

  notch_features = []
  durations = []
  heights = []

  for guy in range(len(df.age)):
    print(guy)
    P, D, H = dgf.features_from_dicrotic_notch(df.time[guy], df.signal[guy])

    notch_features.append(P)
    durations.append(D)
    heights.append(H)

  print("ended")

  print("Extracting features from Guassian fits...")
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

  # =============================================================================
  # add new features
  # =============================================================================

  df['dist_mea'] = dist_mea
  df['dist_std'] = dist_std
  df['dist_med'] = dist_med
  df['dist_mad'] = dist_mad

  df['diff_mea'] = diff_mea
  df['diff_std'] = diff_std
  df['diff_med'] = diff_med
  df['diff_mad'] = diff_mad

  df['n_dist_mea'] = n_dist_mea
  df['n_dist_std'] = n_dist_std
  df['n_dist_med'] = n_dist_med
  df['n_dist_mad'] = n_dist_mad

  df['n_diff_mea'] = n_diff_mea
  df['n_diff_std'] = n_diff_std
  df['n_diff_med'] = n_diff_med
  df['n_diff_mad'] = n_diff_mad

  df['dist_mea_all'] = dist_mea_all
  df['dist_std_all'] = dist_std_all
  df['dist_med_all'] = dist_med_all
  df['dist_mad_all'] = dist_mad_all

  df['diff_mea_all'] = diff_mea_all
  df['diff_std_all'] = diff_std_all
  df['diff_med_all'] = diff_med_all
  df['diff_mad_all'] = diff_mad_all

  df['n_dist_mea_all'] = n_dist_mea_all
  df['n_dist_std_all'] = n_dist_std_all
  df['n_dist_med_all'] = n_dist_med_all
  df['n_dist_mad_all'] = n_dist_mad_all

  df['n_diff_mea_all'] = n_diff_mea_all
  df['n_diff_std_all'] = n_diff_std_all
  df['n_diff_med_all'] = n_diff_med_all
  df['n_diff_mad_all'] = n_diff_mad_all

  # save db
  print("Saving database...")
  df.T.to_json(output_json)
