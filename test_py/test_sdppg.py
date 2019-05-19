# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append("../py")
import sdppg_features
import pytest


def test_find_first_index_for_maximum_in_zero_crossing():
    # test empty zero crossing
    t = np.linspace(0, 2*np.pi, 600)
    start = sdppg_features.find_first_index_for_maximum_in_zero_crossing(
            t, [])
    assert start == 0
    s = np.sin(t)
    # 2PI is the period. We expect start to be 0
    z = [0, int(len(s)/2), len(s)-1]
    start = sdppg_features.find_first_index_for_maximum_in_zero_crossing(
            s, z)
    assert start == 0


def test_find_next_a():
    x = np.linspace(3, 0, 600)
    t = np.linspace(0, 8*np.pi, 600)
    s = np.sin(t)*x
    z = [0] + [int(len(s)/i) for i in np.arange(8, 0, -1)]
    a, k = sdppg_features.find_next_a(s, z, 0)
    assert a == np.max(s)
    assert k == 1


def test_features_from_sdppg():
    x = np.linspace(3, 0, 600)
    t = np.linspace(0, 8*np.pi, 600)
    s = np.sin(t)*x
    q = 100
    sdppg, f = sdppg_features.features_from_sdppg(t, s, normalise=True, f=100)
    assert len(sdppg) == q*len(s)
    assert np.isclose(max(abs(sdppg)), 1, atol=1e-16)
    assert len(f) == 5
    for _, __ in zip(list(f)[:], list(f)[1:]):
        assert len(f[_]) == len(f[__])
    

def test_features_from_sdppg_non_def_param():
    x = np.linspace(3, 0, 600)
    t = np.linspace(0, 8*np.pi, 600)
    s = np.sin(t)*x
    q = 100
    sdppg, f = sdppg_features.features_from_sdppg(t, s, normalise=False, 
                                                  spline=False)
    assert len(sdppg) == len(s)
    assert np.isclose(max(abs(sdppg)), 1) == False
