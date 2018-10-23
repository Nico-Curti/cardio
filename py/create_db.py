#!/usr/bin/python
import sys, argparse
from os.path import basename, join
import glob
import pandas as pd
import numpy as np
from scipy import interpolate, fftpack, stats
from sklearn.metrics import mutual_info_score # mutual information algorithm
from sklearn.metrics.pairwise import euclidean_distances as dist
from statsmodels.tsa.tsatools import lagmat
import json

import pre_process as pr

def detect_peaks(signal, mov_avg):
  window = []
  peaklist = []
  for (i, datapoint), roll in zip(enumerate(signal), mov_avg):
    if (datapoint <= roll) and (len(window) <= 1):
      continue
    elif (datapoint > roll):
      window.append(datapoint)
    else:
      beatposition = i - len(window) + np.argmax(window)
      peaklist.append(beatposition)
      window = []
  return peaklist, [signal[x] for x in peaklist]

calc_RR = lambda peaklist, fs: (np.diff(peaklist) / fs) * 1e3

mutual_info = lambda x, y, bins : mutual_info_score(None,
                                                    None,
                                                    contingency=np.histogram2d(x, y, bins)[0])

MI = lambda x, tmax : [mutual_info(x[i + 1 :], x[: -(i + 1)], 100) for i in range(tmax - 1)]


def fnn(data, m):
  """
  https://github.com/jcvasquezc/Corr_Dim
  Compute the embedding dimension of a time series data to build the phase space using the false neighbors criterion
  data--> time series
  m   --> maximmum embeding dimension
  """
  RT = 15.0
  AT = 2
  sigmay = np.std(data, ddof=1)
  nyr = len(data)
  EM = lagmat(data, maxlag = m - 1)
  EEM = np.asarray([EM[j,:] for j in range(m - 1, EM.shape[0])])
  embedm = m
  for k in range(AT, EEM.shape[1] + 1):
    fnn1 = []
    fnn2 = []
    Ma = EEM[:,range(k)]
    D = dist(Ma)
    for i in range(1, EEM.shape[0] - m - k):
      d = D[i,:]
      pdnz = np.where(d>0)
      dnz = d[pdnz]
      Rm = np.min(dnz)
      l = np.where(d==Rm)
      l = l[0]
      l = l[len(l) - 1]
      if l + m + k - 1 < nyr:
        fnn1.append(np.abs(data[i + m + k - 1] - data[l + m + k - 1]) / Rm)
        fnn2.append(np.abs(data[i + m + k - 1] - data[l + m + k - 1]) / sigmay)
    Ind1=np.where(np.asarray(fnn1) > RT)
    Ind2=np.where(np.asarray(fnn2) > AT)
    if len(Ind1[0]) / len(fnn1) < .1 and len(Ind2[0])/len(fnn2) < .1:
      embedm = k
      break
  return embedm

def create_db(data_dir, info_dir=''):

  features = {}

  datas = sorted(glob.glob(join(data_dir, '*_data.txt')))
  infos = sorted(glob.glob(join(info_dir, '*_info.txt')))

  for f, i in zip(datas, infos):
    bf, bi = basename(f).split('_'), basename(i).split('_')
    assert(bf[1] == bi[1])
    print("Processing file %s"%(basename(f)))

    info = pd.read_csv(i, sep=",", skiprows = 1, names = [0, 1],
                        index_col = 0
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

    data = pd.read_csv(f, sep=',')
    time, sign = pr.process_pipe(data, view=False, output='')
    srate = len(time)/max(time)

    mt, mov_avg = pr.m_avg(time, sign, int(srate*.5))#rolmean(sign, .5, srate)
    peaklist, sign_peak = detect_peaks(sign, mov_avg)

    # compute some common measurements
    RR = calc_RR(peaklist, srate)
#    RR = scipy.signal.detrend(RR, type='linear')
    RR_diff = np.abs(np.diff(RR))
    ibi = np.mean(RR) # mean Inter Beat Interval
    bpm = 60000 / ibi
    sdnn = np.std(RR) # Take standard deviation of all R-R intervals
    sdsd = np.std(RR_diff) # #Take standard deviation of the differences between all subsequent R-R intervals
    rmssd = np.sqrt(np.mean(RR_diff**2)) # #Take root of the mean of the list of squared differences

    pnn20 = len(RR_diff[RR_diff > 20.] / len(RR_diff))
    pnn50 = len(RR_diff[RR_diff > 50.] / len(RR_diff))

    mad = np.median(np.abs(RR - np.median(RR)))

    # compute frequency domain measurements
    # TOFIX
    new_range = np.linspace(0, len(RR), len(RR)*100)
    tck = interpolate.splrep(np.arange(0, len(RR)), RR, s=0)
    RR_new = interpolate.splev(new_range, tck, der=0)
    RR_fft = fftpack.fft(RR_new)
    freq = fftpack.fftfreq(len(RR_new), d=((1./srate)))[:len(RR_new)//2]
    RR_fft = RR_fft[:len(RR_new)//2] / len(RR_new)

    lf = np.trapz(abs(RR_fft[(freq >= .04) & (freq <= .15)]))
    hf = np.trapz(abs(RR_fft[(freq >= .15) & (freq <= .4 )]))

    # other features
    p_data = pd.Series(data=sign).value_counts() / len(sign)
    entropy = stats.entropy(p_data)

    tau_max = 1000
    mi = MI(sign, tau_max)
    mi_t, mi_avg = pr.m_avg(np.arange(0, len(mi)), mi, 100)
    opt_delay = mt[np.argmin(mi_avg)]
    embedim = fnn(sign, 20)

    features[f] = {'sex' : info.Sex.values.item(),
                   'age' : info.Age.values.item(),
                   'weight' : info.Weight.values.item(),
                   'smoke': info.Smoking.values.item(),
                   'afib': info.Afib.values.item(),
                   'rhythm' : info.Rhythm.values.item(),
                   'RR' : RR,
                   'bpm' : bpm,
                   'ibi' : ibi,
                   'sdnn' : sdnn,
                   'sdsd' : sdsd,
                   'rmssd' : rmssd,
                   'pnn20' : pnn20,
                   'pnn50' : pnn50,
                   'mad' : mad,
                   'lf' : lf,
                   'hf' : hf,
                   'entropy' : entropy,
                   'opt_delay' : opt_delay,
                   'embedim' : embedim,
                   'time' : time,
                   'signal' : sign
                   }


  with open('cardio.json', 'w') as f:
    json.dump(features, f)

if __name__ == '__main__':
  description = "Create DB data"
  parser = argparse.ArgumentParser(description = description)
  parser.add_argument("-f", required=True,  dest="data_dir", action="store", help="Data directory name", default='')
  parser.add_argument("-i", required=False, dest="info_dir", action="store", help="Info directory name", default='')

  if len(sys.argv) <= 1:
    parser.print_help()
    sys.exit(1)
  else:  args = parser.parse_args()

  data_dir = args.data_dir
  if args.info_dir == '': info_dir = data_dir
  else:                   info_dir = args.info_dir

  create_db(data_dir, info_dir)

