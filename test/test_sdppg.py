# -*- coding: utf-8 -*-
import numpy as np
from cardio.modules import sdppg_features


def test_find_first_index_for_maximum_in_zero_crossing_empty_zc():
  # test empty zero crossing
  t = np.linspace(0, 2*np.pi, 600)
  start = sdppg_features.find_first_index_for_maximum_in_zero_crossing(
          t, [])
  assert start == 0


def test_find_first_index_for_maximum_in_zero_crossing_with_zc():
  t = np.linspace(-2.*np.pi -0.01, 2*np.pi, 1000)
  s = np.sin(t)
  # 2PI is the period
  d2s = np.gradient(np.gradient(s, t), t)
  z = np.asarray([0] + list(np.where(d2s[0:-1]*d2s[1:] < 0.)[0]))
  start = sdppg_features.find_first_index_for_maximum_in_zero_crossing(
          s, z)
  assert start == 0


def test_find_next_a():
  # simulating a PPG signal as a sum of sinusoidal waves with frequences 1, 2, and 3 Hz
  t = np.linspace(-0.3, 1., 600)
  w = 2*np.pi
  v = np.asarray([1., 2., 3.])
  s = 0
  for f in v:
    s += f**(-2)*np.sin(w*f*t)
  sdppg = np.gradient(np.gradient(s, t), t)
  fdppg = np.gradient(np.gradient(sdppg, t), t)
  zero_crossing = np.where(fdppg[0:-1]*fdppg[1:] < 0.)[0]
  start = sdppg_features.find_first_index_for_maximum_in_zero_crossing(sdppg, zero_crossing)
  a, k = sdppg_features.find_next_a(sdppg, zero_crossing, start)
  assert a == np.max(sdppg[:int(len(t)*.5)])  # we are certain we have a within the first half of our mock sdppg points
  assert k == 2


def test_features_from_sdppg():
  # simulating a PPG signal as a sum of sinusoidal waves with frequences 1, 2, and 3 Hz
  t = np.linspace(-0.3, 1., 600)
  w = 2*np.pi
  v = np.asarray([1., 2., 3.])
  s = 0
  for f in v:
    s += f**(-2)*np.sin(w*f*t)
  q = 100
  sdppg, f = sdppg_features.features_from_sdppg(t, s, normalise=True,
                                                flip=False, spline=True, f=q)
  assert len(sdppg) == q*len(s)
  assert np.isclose(max(abs(sdppg)), 1, atol=1e-16)


def test_dictionary_returned_from_features_from_sdppg():
  t = np.linspace(-0.3, 1., 600)
  w = 2*np.pi
  v = np.asarray([1., 2., 3.])
  s = 0
  for f in v:
    s += f**(-2)*np.sin(w*f*t)
  q = 100
  sdppg, f = sdppg_features.features_from_sdppg(t, s, normalise=True,
                                                flip=False, spline=True, f=q)
  assert len(f) == 10  # "a", "b", "c", "d", "e", "AGI", "t_ab", "t_bc", "t_cd", "t_de"
  for _, __ in zip(list(f)[:], list(f)[1:]):
    assert len(f[_]) == len(f[__])
  keys = ['a', 'b', 'c', 'd', 'e', 'AGI', 't_ab', 't_bc', 't_cd', 't_de']
  assert keys == list(f.keys())


def test_features_from_sdppg_non_normalised():
  t = np.linspace(-0.3, 1., 600)
  w = 2*np.pi
  v = np.asarray([1., 2., 3.])
  s = 0
  for f in v:
    s += f**(-2)*np.sin(w*f*t)
  sdppg, f = sdppg_features.features_from_sdppg(t, s, normalise=False,
                                                flip=False, spline=False)
  assert len(sdppg) == len(s)
  assert np.isclose(max(abs(sdppg)), 1) == False

def test_features_from_sdppg_b_res():
  t = np.linspace(-0.3, .5, 600)
  w = 2*np.pi
  v = np.asarray([1., 2., 3.])
  s = 0
  for f in v:
    s += f**(-2)*np.sin(w*f*t)
  sdppg, f = sdppg_features.features_from_sdppg(t, s, normalise=False,
                                                spline=False)
  for _ in f['b']:
    assert _ <= 0

def test_sdppg_agi():
  a = [1, .8, .3]
  b = [-.9, -.7, -.2]
  c = [.1, .1, .1]
  d = [-.1, -.1, -.1]
  e = c
  # test float
  agi_single = sdppg_features.sdppg_agi(a[0], b[0], c[0], d[0], e[0])
  assert np.isclose(agi_single, -1, atol=1e-16)
  # test array-like
  agi_array = sdppg_features.sdppg_agi(a, b, c, d, e)
  for _ in agi_array:
    assert np.isclose(_, -1, atol=1e-16)
