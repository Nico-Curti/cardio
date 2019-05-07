#!/usr/bin/python

# References: https://github.com/paulvangentcom/heartrate_analysis_python

import sys, argparse
from os.path import basename, join
import glob
import pandas as pd
import numpy as np
from scipy.special import entr
from scipy.stats import skew, kurtosis, pearsonr  # , skewtest, kurtosistest
from scipy import interpolate, fftpack
from sklearn.metrics import mutual_info_score # mutual information algorithm
from sklearn.metrics.pairwise import euclidean_distances as dist
from statsmodels.tsa.tsatools import lagmat
import json
import matplotlib.pylab as plt
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


mutual_info = lambda x, y, bins : mutual_info_score(None,
                                                    None,
                                                    contingency=np.histogram2d(x, y, bins)[0])

MI = lambda x, tmax : [mutual_info(x[i + 1 :], x[: -(i + 1)], 100) for i in range(tmax - 1)]
mad = lambda x, medianX : np.median(np.abs(x-medianX))

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


def create_db(data_dir, info_dir=""):
  features = {}

  datas = sorted(glob.glob(join(data_dir, "*_data.txt")))
  infos = sorted(glob.glob(join(info_dir, "*_info.txt")))

  for f, i in zip(datas[:], infos[:]):
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
    time, sign = pr.process_pipe(data, view=False, output='', name=bf[1])

    srate = len(time)/(max(time)-min(time))  #sampling rate
    peaks = np.arange(len(sign))  #initializing peaks index

    # making peak detection using 4 different sliding windows, their width is in srate units
    for i in np.array([.5, 1., 1.5, 2.]):  # widths
      mt, mov_avg = pr.m_avg(time, sign, int(srate*i))
      len_filler = np.zeros((len(sign)-len(mov_avg))//2) + np.mean(sign) # used to make mov_avg the same size as sign
      mov_avg = np.insert(mov_avg, 0, len_filler)
      mov_avg = np.append(mov_avg, len_filler)

      peaklist, sign_peak = detect_peaks(sign,mov_avg)
      peaks = np.intersect1d(peaks, peaklist)  # keeping only peaks detected with all 4 different windows


		# peaks' checking: rejecting lower peaks where RR distance is too small
    final_peaks = [] # definitive peaks positions container
    last_peak = -1  # control parameter to avoid undesired peaks still in final_peaks
    for p in peaks:
      if p <= last_peak:
        continue
      evaluated_peaks = [g for g in peaks if p <= g <= srate * .5 + p ]  # peaks evaluated t once, only 1 of them will be kept in final_peaks
      last_peak = max(evaluated_peaks)
      final_peaks.append(evaluated_peaks[np.argmax([sign[x] for x in evaluated_peaks])])

    final_peaks = np.unique(final_peaks)  # to avoid repetition of identical elements


		# computation of quality coefficient
    grad = np.gradient(sign)  # gradient of signal
    checker = np.multiply(grad[:-1], grad[1:]) # equals to grad_i * grad_i+1

    # to count "how many times signal inverts its growth without having a peak" (index of bad quality for our signals)
    # we can count how many times gradient changes its sign (local maxima or local minima) and subtract twice the number of peaks to that
    # then we can multiply by the variance of peaks' amplitude to enhance the bad contribute given by messy peaks

    quality = np.var([sign[x] for x in final_peaks])*(len(checker[checker < 0]) - 2 * len(final_peaks) + 2) / len(checker)  # "+2" is just to avoid negative numbers

    # Note Well: now quality it's a defect index. It goes from 0 to 1
    # where 0 is for perfect signals and >0.05 is for horrible signals.
    # usually we have (with some exceptions):
    # quality ~ 1e-5 or less  PERFECT
    # quality ~ 1e-4          GOOD
    # quality ~ 1e-3          GOOD WITH PROBLEMS
    # quality ~ 2e-2          BAD
    # quality > 5e-2          HORRIBLE


    # compute some common measurements
    RR = np.diff(time[final_peaks], 1, 0,)*1e3  # time between consecutive R peaks (in milliseconds)
    RR_diff = np.abs(np.diff(RR, 1, 0))  # time variation between consecutive RR intervals
    ibi = np.mean(RR)  # mean Inter Beat Interval
    bpm = 60000 / ibi  # mean bpm
    sdnn = np.std(RR)  # Take standard deviation of all R-R intervals
    sdsd = np.std(RR_diff)  # #Take standard deviation of the differences between all subsequent R-R intervals
    rmssd = np.sqrt(np.mean(RR_diff**2))  #Take the square root of the mean of the list of squared differences

    x = np.multiply(RR[1:-1]-RR[:-2], RR[1:-1]-RR[2:]) # we have a turning point when x>0
    turning_point_ratio = lambda w : len(w[w>0.]) / len(w) # turning point ratio (randomness index)
    tpr = turning_point_ratio(x)
    x = np.multiply(RR_diff[1:-1]-RR_diff[:-2], RR_diff[1:-1]-RR_diff[2:]) #same but with RR_diff instead of RR
    tpr_RR_diff = turning_point_ratio(x)
    pnn20 = len(RR_diff[RR_diff > 20.]) / len(RR_diff)  # percentage of RR_diff > 20 milliseconds
    pnn50 = len(RR_diff[RR_diff > 50.]) / len(RR_diff)  # percentage of RR_diff > 50 milliseconds

    medianRR = np.median(RR)
    madRR = mad(RR, medianRR)
    
    Rpeakvalues = sign[final_peaks]
    AA = np.diff(Rpeakvalues, n=1, axis=0)
    medianAA = np.median(AA)
    madAA = mad(AA, medianAA)

    # compute frequency domain measurements
    new_range = np.linspace(0, len(RR), len(RR)*100)  # resampling data at higher frequency (x100)
    tck = interpolate.splrep(np.arange(0, len(RR)), RR, s=0)
    RR_new = interpolate.splev(new_range, tck, der=0)
    RR_fft = fftpack.fft(RR_new)  # fast fourier transform
    freq = fftpack.fftfreq(len(RR_new), d=(1./srate))[:len(RR_new)//2]
    RR_fft = RR_fft[:len(RR_new)//2] / len(new_range)

    lf = np.trapz(abs(RR_fft[(freq >= .04) & (freq <= .15)]))  # Low Frequency is related to short-term blood pressure regulation
    hf = np.trapz(abs(RR_fft[(freq > .15) & (freq <= .4 )]))  # High Frequency is related to breathing

    # other features
    entropyRR = sum(entr(RR/sum(RR)))  # signal special-entropy

    tau_max = 400  #  keep an eye on. probably must be < 439 (file 1380 is the shortest)
    mi = MI(sign, tau_max)  # mutual information
    mi_t, mi_avg = pr.m_avg(np.arange(0, len(mi)), mi, 100)
    opt_delay = mt[np.argmin(mi_avg)]
    embedim = fnn(sign, 20)
    skewRR = skew(RR)
    kurtRR = kurtosis(RR)
    skewAA = skew(AA)
    kurtAA = kurtosis(AA)
    meanAA = np.mean(AA)
    stddevAA = np.std(AA)
    corrRRAA, twotailedPvalue_corrRRAA = pearsonr(RR, AA)

    features[f] = {"sex" : info.Sex.values.item(),
                   "age" : info.Age.values.item(),
                   "weight" : info.Weight.values.item(),
                   "smoke": info.Smoking.values.item(),
                   "afib": info.Afib.values.item(),
                   "rhythm" : info.Rhythm.values.item(),
                   "quality" : quality,
                   "RR" : RR.tolist(),
                   "bpm" : bpm,
                   "ibi" : ibi,
                   "sdnn" : sdnn,
                   "sdsd" : sdsd,
                   "rmssd" : rmssd,
                   "tpr" : tpr,
                   "tpr_RR_diff" : tpr_RR_diff,
                   "pnn20" : pnn20,
                   "pnn50" : pnn50,
                   "medianRR" : medianRR,
                   "madRR" : madRR,
                   "lf" : lf,
                   "hf" : hf,
                   "skewnessRR" : skewRR,
                   "kurtosisRR" : kurtRR, 
                   "entropyRR" : entropyRR,
                   "opt_delay" : opt_delay,
                   "embedim" : embedim,
                   "time" : time.tolist(),
                   "signal" : sign.tolist(),
                   "Rpeakvalues" : Rpeakvalues.tolist(),
                   "AA" : AA.tolist(),
                   "medianAA" : medianAA,
                   "madAA" : madAA,
                   "skewnessAA" : skewAA,
                   "kurtosisAA" : kurtAA, 
                   "meanAA" : meanAA,
                   "std_devAA" : stddevAA,
                   "corrRRAA" : corrRRAA,
                   "corrRRAA_Pvalue" : twotailedPvalue_corrRRAA
                   }

  # saving data on file cardio.json
  with open('cardio.json', 'w') as file:
    json.dump(features, file)

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
