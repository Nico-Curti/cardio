
# -*- coding: utf-8 -*-

# cardio stuff
import clean_db
import create_db
import pre_process as pr
import double_gaussian_features as dgf
import sdppg_features


# standard libraries
import numpy as np
from scipy.optimize import curve_fit
from operator import itemgetter
import pandas as pd
import json
import glob
from os.path import basename, join

# %%
# load information from info dir
dd = {}
data_dir = "../data/all/"
info_dir = data_dir

# %%
data = sorted(glob.glob(join(data_dir, "*_data.txt")))
info = sorted(glob.glob(join(info_dir, "*_info.txt")))
for f, i in zip(data[:], info[:]):
  bf, bi = basename(f).split('_'), basename(i).split('_')
  assert(bf[1] == bi[1])
  print("Processing file %s"%(basename(f)))

  series_info = pd.read_csv(i, sep=",", skiprows = 1, names = [0, 1],
                      index_col = 0, encoding = 'latin1',
                      # Smoking
                      ).T.replace("NO", 0
                      ).replace("YES", 1
                      # Sex
                      ).replace("Man", "M"
                      ).replace("Vrouw", "F"
                      ).replace("Female", "F"
                      ).replace("Male", "M"
                      ).replace("(null)", np.nan
                      # Rhythm
                      ).replace("Onbekend", 0 # sconosciuto
                      ).replace("Unknown", 0
                      ).replace("Boezemfibrilleren", 1 # atrial fib
                      ).replace("Atrial Fibrillation", 1
                      ).replace("Atrial Flutter", 2
                      ).replace("AVNRT", 3 # Tachicardia da rientro atrio-ventricolare di tipo nodale
                      ).replace("Boezemtachycardie", 4 # Tachicardia emotiva
                      ).replace("Atrial Tachycardia", 5 # Tachicardia atriale
                      ).replace("Extrasystolen in de boezems", 6 # Extrasistole negli atri
                      ).replace("Extrasystolen in de kamers", 7 # Extrasystole nelle stanze
                      ).replace("Boezemflutter", 8 # Bosom flutter
                      ).replace("VES", 9 # velocità di eritrosedimentazione (infiammazione)
                      )
  dd[f] = {'length': series_info.Length.values.item(),
           'city': series_info.City.values.item(),
           'country': series_info.Country.values.item(),
           'lifestyle': series_info.Lifestyle.values.item(),
           'class': series_info.Class.values.item()  # ,
           #'filename': series_info.Filename.values.item(),  # useless
           #'filenumber': series_info.Filenumber.values.item()  # useless
      }

with open('cardio_new_info.json', 'w') as file:
  json.dump(dd, file)

# %%
# load json files
f = "../data/cardio_all.json"
df_old = pd.DataFrame(json.load(open(f)))
f_info = "cardio_new_info.json"
df_info = pd.DataFrame(json.load(open(f_info)))
# %%
# merge dataframes
df_info_t = df_info.T
merged = df_old.T
for __ in df_info_t.keys():
  merged[__] = df_info_t[__].to_numpy()
# %%
# save merged file dataframes
merged_dict = merged.T.to_dict()
f = '../cardio_merged.json'
with open(f, 'w') as file:
  json.dump(merged_dict, file)

# %%
df = clean_db.db_to_dataframe(f)

# %%

# get sdppg features
features = []

for _, __ in zip(df.time, df.signal):
  sdppg, ff = sdppg_features.features_from_sdppg(_, __, spline=False, normalise=False)
  features.append(ff)

# %%

# add features to dataframe
keys = list(features[0].keys())

for key in keys:
  p = []
  for _ in features:
    p.append(np.median(_[key]))
  df[key] = p
# %%

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

# %%

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


# %%
# length is in cm
df['bmi'] = df['weight'].astype(float)/((df['length'].astype(float))**2/1E4)

# %%
df_dict = df.T.to_dict()

# %%
with open('../cardio_new.json', 'w') as file:
  json.dump(df_dict, file)

# %%
df_new = clean_db.db_to_dataframe('../cardio_new.json')


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


# =============================================================================
# EXTRACT FEATURES
# =============================================================================

notch_features = []
durations = []
heights = []
x_of_max = []

for guy in range(len(db.age)):
  print(guy)
  P, D, H, X_M = dgf.features_from_dicrotic_notch(db.time[guy], db.signal[guy])

  notch_features.append(P)
  durations.append(D)
  heights.append(H)
  x_of_max.append(X_M)
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

db.T.to_json('./final_cardio.json')
