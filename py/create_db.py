#!/usr/bin/python
import sys, argparse
from os.path import basename, join
import glob
import pandas as pd
import numpy as np
import pre_process as pr
import json

def rolmean(signal, hrw, fs):
  dataset = pd.Series(data=signal)
  mov_avg = dataset.rolling(int(hrw * fs)).mean()
  avg_hr  = np.mean(signal)
  mov_avg = np.asarray([avg_hr if np.isnan(x) else x for x in mov_avg])
  return mov_avg * 1.2

def detect_peaks(signal, mov_avg):
  window = []
  peaklist = []
  for (i, datapoint), roll in zip(enumerate(signal), mov_avg):
    if (datapoint < roll) and (len(window) < 1):
      continue
    elif (datapoint > roll):
      window.append(datapoint)
    else:
      beatposition = i - len(window) + np.argmax(window)
      peaklist.append(beatposition)
      window = []
  return peaklist, [signal[x] for x in peaklist]


calc_RR = lambda peaklist, fs: (np.diff(peaklist) / fs) * 1e3


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
                        ).replace("Boezemfibrilleren", 0 # atrial fib
                        ).replace("AVNRT", 1 # Tachicardia da rientro atrio-ventricolare di tipo nodale
                        ).replace("Onbekend", np.nan # sconosciuto
                        ).replace("Atrial Fibrillation", 2
                        ).replace("Boezemtachycardie", 3 # Tachicardia emotiva
                        ).replace("Extrasystolen in de boezems", 4 # Extrasistoli negli atri
                        ).replace("Boezemflutter", 5 # Bosom flutter
                        ).replace("Extrasystolen in de kamers", 6 # Extrasystolen nelle stanze
                        ).replace("Unknown", np.nan
                        ).replace("Atrial Tachycardia", 7
                        ).replace("Atrial Flutter", 8
                        ).replace("VES", 9 # velocità di eritrosedimentazione (infiammazione)
                        )

    data = pd.read_csv(f, sep=',')
    time, sign = pr.process_pipe(data, view=False, output='')
    srate = len(time)/max(time)

    mov_avg = rolmean(sign, .5, srate)
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
    # TODO


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

