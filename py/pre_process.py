#!/usr/bin/python

# parse command line
import sys, argparse
from os.path import basename, splitext
# pre processing data
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import interpolate, signal

guard = 1000 # number of points to append before filtering (artifacts remove)
new_freq = 180 # interpolatin freq (Hz)
startbpm = 40
limitbpm = 240
order = 4     # filter order
t_avg = .1    # time interval for averaging (sec)
n_mov_avg = int(np.ceil(t_avg * .5 * new_freq))
f1 = startbpm / 60; # Start Frequency
f2 = limitbpm / 60; # Cut off Frequency
Wn = np.asarray([f1, f2]) / (new_freq * .5) # Butterworth paramter

def interpolation(data):
  t_int   = np.arange(data.Time.iloc[0], data.Time.iloc[-1], 1./new_freq)
  tck     = interpolate.splrep(data.Time, data.R, s=0)
  red_int = interpolate.splev(t_int, tck, der=0)

  return t_int, red_int

filtering = lambda x    : signal.lfilter(*signal.butter(order, Wn, "bandpass"), x)
m_avg     = lambda t, x : (np.asarray([t[i] for i in range(n_mov_avg, len(x) - n_mov_avg)]),
                           np.convolve(x, np.ones((2*n_mov_avg + 1, )) / (2*n_mov_avg + 1), mode = 'valid'))

def process_pipe(data, view = False, output = ""):
  # spline interpolation of data
  t, r = interpolation(data)

  r = np.insert(r, 0, r[:guard])
  # filter data
  rf = filtering(r)
  rf = rf[guard:]
#  rf *= -1 # ecg fmt of the curve
  # moving average
  tma, rma = m_avg(t, rf)

  if view:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, figsize = (10, 8), sharex=True)
    ax1.plot(t, r[guard:], "b-", label="Spline")
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(.8, 1)
    ax1.set_title("Raw", fontsize=14, fontweight="bold")

    ax2.plot(t, rf, 'r-', label="Butterworth filter")
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(-.05, .05)
    ax2.set_title("Raw -> filtered", fontsize=14, fontweight="bold")

    ax3.plot(tma, rma, "g-", label="Move Avg")
    ax3.set_xlim(0, t[-1])
    ax3.set_ylim(-.05, .05)
    ax3.set_title("Raw -> filtered -> move avg", fontsize=14, fontweight="bold")

    ax3.set_xlabel("Time (sec)", fontsize=14) # common axis label
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')
  return tma, rma


if __name__ == '__main__':
  description = "Pre-Processing data"
  parser = argparse.ArgumentParser(description = description)
  parser.add_argument("-f", required=True,  dest="data", action="store", help="Data filename", default="")
  parser.add_argument("-v", required=False, dest="view", action="store", help="View results",  default=False)

  if len(sys.argv) <= 1:
    parser.print_help()
    sys.exit(1)
  else:  args = parser.parse_args()

  filename = args.data
  view = bool(args.view)

  base = basename(filename)
  f, _ = splitext(base)
  output = f + '.png'

  data = pd.read_csv(filename, sep=",")
  process_pipe(data, view, output)


