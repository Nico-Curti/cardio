# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd

def db_to_dataframe(filename):
  """
  Load the json file obtained through create_db into a pandas dataframe

  Parameters
  ----
  filename: string; the full path to the json file, complete with extension

  Return
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
  """
  d = json.load(open(filename))
  d = pd.DataFrame(d).T
  d = d.set_index(np.arange(len(d)))
  return d


def clean_db(db, drop_columns=['rhythm'],
             drop_na_in=['weight', 'tpr', 'madRR', 'medianRR', 'opt_delay',
                      'afib', 'age', 'sex'],
              drop_zeroes_in=['weight', 'age'],
              quality_threshold=None,
              reset_index=True
              ):
  """
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
  >>>  d[name] = {'rhythm': np.nan, 'weight': _, 'age': __}
  >>>d = pd.DataFrame(d).T
  >>>clean_db(d)
    Out:
        age  weight
    0  40.0    56.0
    1  56.0    66.0
  """
  newdb = db
  if quality_threshold is not None:
    newdb = newdb[newdb['quality'] < quality_threshold]
  if drop_columns is not None:
    dcn = list(set(drop_columns) & set(newdb.columns))
    if len(dcn) != 0:
      newdb = newdb.drop(columns=dcn)
  if drop_na_in is not None:
    dnn = list(set(drop_na_in) & set(newdb.columns))
    if len(dnn) != 0:
      newdb = newdb.dropna(subset=dnn)
  if drop_zeroes_in is not None:
    dzn = list(set(drop_zeroes_in) & set(newdb.columns))
    if len(dzn) != 0:
      for _ in dzn:
        newdb = newdb[newdb[_].astype(float) > 0]
  if reset_index == True:
    newdb = newdb.set_index(np.arange(len(newdb)))
  return newdb
