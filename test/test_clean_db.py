# -*- coding: utf-8 -*-
import numpy as np
import json
import pandas as pd
import os
from cardio.modules import clean_db

def test_db_to_dataframe():
  f = {}
  filename = "f.json"
  with open(filename, 'w') as file:
    json.dump(f, file)
  d = clean_db.db_to_dataframe(filename)
  assert isinstance(d, pd.DataFrame)
  assert d.empty
  os.remove(filename)


def test_clean_db_optionalNone():
  d = pd.DataFrame()
  N = 100
  d['quality'] = np.linspace(0, 1E-3, 5*N)
  d['rhythm'] = np.asarray([1, 2, 3, 4, np.nan]*N)
  d['weight'] = np.asarray(np.asarray([0,]*N + list(np.arange(N))*4))
  d['age'] = d['weight']
  df = clean_db.clean_db(d, drop_columns=None, drop_na_in=None,
                         drop_zeroes_in=None, quality_threshold=None)
  assert len(df) == len(d)


def test_clean_db_quality():
  d = pd.DataFrame()
  N = 100
  d['quality'] = np.linspace(0, 1E-3, 5*N)
  d['rhythm'] = np.asarray([1, 2, 3, 4, np.nan]*N)
  d['weight'] = np.asarray(np.asarray([0,]*N + list(np.arange(N))*4))
  d['age'] = d['weight']
  df = clean_db.clean_db(d, drop_columns=None, drop_na_in=None,
                         drop_zeroes_in=None, quality_threshold=1E-4)
  assert max(df.quality) < 1E-4
  assert len(df) < len(d)
  assert len(df) == sum(np.where(d.quality < 1E-4, 1, 0))


def test_clean_db_drop_columns():
  d = pd.DataFrame()
  N = 100
  d['quality'] = np.linspace(0, 1E-3, 5*N)
  d['rhythm'] = np.asarray([1, 2, 3, 4, np.nan]*N)
  d['weight'] = np.asarray(np.asarray([0,]*N + list(np.arange(N))*4))
  d['age'] = d['weight']
  drop_columns_df=['rhythm', 'weight']
  df = clean_db.clean_db(d, drop_columns=drop_columns_df,
                          drop_na_in=None, drop_zeroes_in=None,
                          quality_threshold=None)
  df_col = list(set(df.columns) & set(drop_columns_df))
  assert len(df_col) == 0
  assert len(df) == len(d)


def test_clean_db_drop_na():
  d = pd.DataFrame()
  N = 100
  d['quality'] = np.linspace(0, 1E-3, 5*N)
  d['rhythm'] = np.asarray([1, 2, 3, 4, np.nan]*N)
  d['weight'] = np.asarray(np.asarray([0,]*N + list(np.arange(N))*4))
  d['age'] = d['weight']
  df = clean_db.clean_db(d, drop_columns=None,
                          drop_na_in=['rhythm'], drop_zeroes_in=None,
                          quality_threshold=None)
  assert len(df) == 4*N
  assert len(df.dropna(subset=('rhythm',))) == len(df)


def test_clean_db_drop_zeroes():
  d = pd.DataFrame()
  N = 100
  d['quality'] = np.linspace(0, 1E-3, 5*N)
  d['rhythm'] = np.asarray([1, 2, 3, 4, np.nan]*N)
  d['weight'] = np.asarray(np.asarray([0,]*N + list(np.arange(N))*4))
  d['age'] = d['weight']
  zeroes_labels = ['age', 'weight']
  df = clean_db.clean_db(d, drop_columns=None,
                          drop_na_in=None, drop_zeroes_in=zeroes_labels,
                          quality_threshold=None)
  assert len(df) == len(d) - 104  # 100 zeroes + 4 zeroes
  for _ in zeroes_labels:
    assert min(abs(df[_])) > 0


