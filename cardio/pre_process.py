#!/usr/bin/python

# parse command line
import sys, argparse
from os.path import basename, splitext
# pre processing data
import numpy as np
import pandas as pd
from scipy import signal#, interpolate


m_avg = lambda t, x, w : (np.asarray([t[i] for i in range(w, len(x) - w)]),
                          np.convolve(x, np.ones((2*w + 1, )) / (2*w + 1),
                                      mode = 'valid'))


def process_pipe(data, view=False, output="", name=""):  
  fs = len(data.Time)/(max(data.Time)-min(data.Time))  # sample rate
  
  # moving average
  w_size = int(fs*.5)  # width of moving window
  mt, ms = m_avg(data.Time, data.R, w_size)  # computation of moving average 
  
  # remove global modulation
  sign = (data.R.iloc[w_size : -w_size] - ms).values

  # compute signal envelope
  analytical_signal = np.abs(signal.hilbert(sign))

  fs = len(sign)/(max(mt)-min(mt))
  w_size = int(fs)
  # moving averate of envelope
  mt_new, mov_avg = m_avg(mt, analytical_signal, w_size)

  # remove envelope
  signal_pure = sign[w_size : -w_size] / mov_avg

  if view:
    
    import matplotlib.pylab as plt

    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, figsize = (10, 8), sharex=True)
    ax1.plot(data.Time, data.R, "b-", label="Original")
    ax1.legend(loc='best')
    ax1.set_title("File "+str(name)+" Raw", fontsize=14)#, fontweight="bold")
    
    ax2.plot(mt, sign, 'r-', label="Pure signal")
    ax2.plot(mt_new, mov_avg, 'b-', label='Modulation', alpha=.5)
    ax2.legend(loc='best')
    ax2.set_title("Raw -> filtered", fontsize=14)#, fontweight="bold")

    ax3.plot(mt_new, signal_pure, "g-", label="Demodulated")
    ax3.set_xlim(0, mt[-1])
    ax3.set_title("Raw -> filtered -> demodulated", fontsize=14)#, fontweight="bold")

    ax3.set_xlabel("Time (sec)", fontsize=14) # common axis label
    ax3.legend(loc='best')
    
    fig.tight_layout()
    plt.savefig(output, bbox_inches='tight')

  return mt_new, signal_pure


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
