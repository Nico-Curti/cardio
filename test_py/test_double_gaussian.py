
# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append("../py")
import double_gaussian_features
import pytest


def test_find_x_of_minima():

  # common parameters in real signals:
  # lenght = 102 seconds, sampling frequency = 30 points per second
  x = np.linspace(0, 102, 3060)

  # simulated signal with dicrotic notches
  y = 2*(np.sin(x)+np.cos(x))+(np.sin(2*x)+np.cos(2*x))

  # we want to find only the minimum of each single period,
  # not all the local minima
  correct_x_of_minima = [141, 330, 518, 707, 895, 1083, 1272, 1460, 1649, 1837,
                         2026, 2214, 2403, 2591, 2779, 2968]

  x_of_minima = double_gaussian_features.find_x_of_minima(x, y)
  assert(len(x_of_minima) == 16)  # because 16 is len(correct_x_of_minima)
  assert(np.all(x_of_minima == correct_x_of_minima))

  # simulated signal with dicrotic notches and significative perturbations
  y = 2*(np.sin(x)+np.cos(x))+(np.sin(2*x)+np.cos(2*x)) + np.sin(x/10)

  # the below list of correct values is slightly different due to perturbation
  # of "+sin(x/10)"
  correct_x_of_minima = [141, 330, 518, 707, 896, 1084, 1272, 1460, 1648,
                         1837, 2025, 2214, 2403, 2591, 2780, 2968]

  x_of_minima = double_gaussian_features.find_x_of_minima(x, y)
  assert(len(x_of_minima) == 16)
  assert(np.all(x_of_minima == correct_x_of_minima))

  # simulated signal without dicrotic notches
  y = np.sin(x)

  correct_x_of_minima = [141, 330, 518, 707, 895, 1083, 1272, 1460, 1649,
                         1837, 2026, 2214, 2403, 2591, 2779, 2968]

  x_of_minima = double_gaussian_features.find_x_of_minima(x, y)
  assert(len(x_of_minima) == 16)
  assert(np.all(x_of_minima == correct_x_of_minima))

  # simulated signal with perturbation without dicrotic notches
  y = np.sin(x) + np.sin(x/10)

  correct_x_of_minima = [139, 328, 519, 709, 898, 1086, 1273, 1460, 1647,
                         1834, 2023, 2213, 2403, 2593, 2782, 2970]

  x_of_minima = double_gaussian_features.find_x_of_minima(x, y)
  assert(len(x_of_minima) == 16)
  assert(np.all(x_of_minima == correct_x_of_minima))


def test_features_from_dicrotic_notch():

  # common parameters in real signals:
  # lenght = 102 seconds, sampling frequency = 30 points per second
  x = np.linspace(0, 102, 3060)

  # simulated signal with dicrotic notches
  y = 2*(np.sin(x)+np.cos(x))+(np.sin(2*x)+np.cos(2*x))
  features = double_gaussian_features.features_from_dicrotic_notch(x, y)
  # Please note that features is a tuple containing 4 elements:
  # a list of 6-element-lists (the inner 6 elements are the fitting parameters)
  # a list of beat durations (max(X)-min(X) done for each single period)
  # a list of beat total height gap (max(Y)-min(Y) done for each single period)
  # a list of x positions of maxima done for each single period

  # 2 different asserts are needed since features[0] is more inner-nested than
  # features element from 1 to 3, so the np.isnan() function raises a TypeError
  assert(not np.any(np.isnan(features[0])))
  assert(not np.any(np.isnan(features[1:])))
  assert(len(features[0]) == len(features[1]) == len(features[2]))
  assert(np.size(features[0]) == len(features[0])*6.)
  assert(np.size(features) == len(features[1])*4.)

  fitting_params = np.asarray(features[0]).T  # to easier access them later
  mean_fitting_params = np.mean(fitting_params, axis=1)
  mean_duration = np.mean(features[1])
  mean_total_gap = np.mean(features[2])
  mean_gaussians_distance = mean_fitting_params[4]-mean_fitting_params[1]
  # last one is the mean distance between gaussian2 and gaussian1

  # variations under 1% of mean value are allowed since we have a discrete
  # periodic signal, so we expect fitting parameters with little variations due
  # to small difference in period duration or in total amplitude gap
  assert(np.all(np.logical_and(mean_duration*.99 < features[1],
                               features[1] < mean_duration*1.01)))
  assert(np.all(np.logical_and(mean_total_gap*.99 < features[2],
                               features[2] < mean_total_gap*1.01)))
  assert(np.all(np.logical_and(mean_fitting_params[0]*.99 < fitting_params[0],
                fitting_params[0] < mean_fitting_params[0]*1.01)))
  assert(np.all(np.logical_and(mean_fitting_params[2]*.99 < fitting_params[2],
                fitting_params[2] < mean_fitting_params[2]*1.01)))
  assert(np.all(np.logical_and(mean_fitting_params[3]*.99 < fitting_params[3],
                fitting_params[3] < mean_fitting_params[3]*1.01)))
  assert(np.all(np.logical_and(mean_fitting_params[5]*.99 < fitting_params[5],
                fitting_params[5] < mean_fitting_params[5]*1.01)))
  assert(np.all(np.logical_and(mean_gaussians_distance*.99
                               < fitting_params[4]-fitting_params[1],
                               fitting_params[4]-fitting_params[1]
                               < mean_gaussians_distance*1.01)))

  # ---------------------------------------------------------------------------
  # simulated signal with dicrotic notches and significative perturbations
  y = 2*(np.sin(x)+np.cos(x))+(np.sin(2*x)+np.cos(2*x)) + np.sin(x/10)
  features = double_gaussian_features.features_from_dicrotic_notch(x, y)

  assert(not np.any(np.isnan(features[0])))
  assert(not np.any(np.isnan(features[1:])))
  assert(len(features[0]) == len(features[1]) == len(features[2]))
  assert(np.size(features[0]) == len(features[0])*6.)
  assert(np.size(features) == len(features[1])*4.)
  # no controls on the values since there's a perturbation due to +np.sin(x/10)

  # ---------------------------------------------------------------------------
  # simulated signal without dicrotic notches
  y = np.sin(x)
  features = double_gaussian_features.features_from_dicrotic_notch(x, y)

  assert(not np.any(np.isnan(features[0])))
  assert(not np.any(np.isnan(features[1:])))
  assert(len(features[0]) == len(features[1]) == len(features[2]))
  assert(np.size(features[0]) == len(features[0])*6.)
  assert(np.size(features) == len(features[1])*4.)

  fitting_params = np.asarray(features[0]).T
  mean_fitting_params = np.mean(fitting_params, axis=1)
  mean_duration = np.mean(features[1])
  mean_total_gap = np.mean(features[2])
  mean_gaussians_distance = mean_fitting_params[4]-mean_fitting_params[1]

  assert(np.all(np.logical_and(mean_duration*.99 < features[1],
                               features[1] < mean_duration*1.01)))
  assert(np.all(np.logical_and(mean_total_gap*.99 < features[2],
                               features[2] < mean_total_gap*1.01)))
  assert(np.all(np.logical_and(mean_fitting_params[0]*.99 < fitting_params[0],
                fitting_params[0] < mean_fitting_params[0]*1.01)))
  assert(np.all(np.logical_and(mean_fitting_params[2]*.99 < fitting_params[2],
                fitting_params[2] < mean_fitting_params[2]*1.01)))
  assert(np.all(np.logical_and(mean_fitting_params[3]*.99 < fitting_params[3],
                fitting_params[3] < mean_fitting_params[3]*1.01)))
  assert(np.all(np.logical_and(mean_fitting_params[5]*.99 < fitting_params[5],
                fitting_params[5] < mean_fitting_params[5]*1.01)))
  assert(np.all(np.logical_and(mean_gaussians_distance*.99
                               < fitting_params[4]-fitting_params[1],
                               fitting_params[4]-fitting_params[1]
                               < mean_gaussians_distance*1.01)))

  # ---------------------------------------------------------------------------
  # simulated signal with perturbation without dicrotic notches
  y = np.sin(x) + np.sin(x/10)
  features = double_gaussian_features.features_from_dicrotic_notch(x, y)

  assert(not np.any(np.isnan(features[0])))
  assert(not np.any(np.isnan(features[1:])))
  assert(len(features[0]) == len(features[1]) == len(features[2]))
  assert(np.size(features[0]) == len(features[0])*6.)
  assert(np.size(features) == len(features[1])*4.)
