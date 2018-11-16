#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:32:44 2018

@author: lyro
"""
import json
from os.path import basename
import numpy as np
from collections import Counter
from operator import itemgetter
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pylab as plt
import sklearn.cluster as skc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from tpot import TPOTClassifier


# loading database file
with open('cardio.json', 'r') as file:
  database = json.load(file)

# containers for specific datas
names = []
sex = []
age = []
weight = []
smoke = []
afib = []
rhythm = []
quality = []
RR = []
bpm = []
ibi = []
sdnn = []
sdsd = []
rmssd = []
tpr = []
tpr_RR_diff = []
pnn20 = []
pnn50 = []
mad = []
lf = []
hf = []
entropy = []
opt_delay = []
embedim = []
time = []
signal = []


#filling the containers with datas (eventually those with better quality)
for d in database.items():
#  if(d[1]['quality']>0.01):  # because quality is a defect index
#    continue

  if(d[1]['afib']==1): continue

  names.append(basename(d[0]).split('_')[1])
  sex.append(d[1]['sex'])
  age.append(d[1]['age'])
  weight.append(d[1]['weight'])
  smoke.append(d[1]['smoke'])
  afib.append(d[1]['afib'])
  rhythm.append(d[1]['rhythm'])
  quality.append(d[1]['quality'])
  RR.append(d[1]['RR'])
  bpm.append(d[1]['bpm'])
  ibi.append(d[1]['ibi'])
  sdnn.append(d[1]['sdnn'])
  sdsd.append(d[1]['sdsd'])
  rmssd.append(d[1]['rmssd'])
  tpr.append(d[1]['tpr'])
  tpr_RR_diff.append(d[1]['tpr_RR_diff'])
  pnn20.append(d[1]['pnn20'])
  pnn50.append(d[1]['pnn50'])
  mad.append(d[1]['mad'])
  lf.append(d[1]['lf'])
  hf.append(d[1]['hf'])
  entropy.append(d[1]['entropy'])
  opt_delay.append(d[1]['opt_delay'])
  embedim.append(d[1]['embedim'])
  time.append(d[1]['time'])
  signal.append(d[1]['signal'])

# choosing datas to work with
datas = np.vstack((quality, bpm, ibi, sdnn, sdsd, rmssd, tpr, tpr_RR_diff,
                   pnn20, pnn50, mad, lf, hf, entropy, opt_delay, embedim)).T

labels = age  # choosing labels to work with
# mean age is 44.869249394673126

#%%
# eventually random undersampling to have even samples per clusters


rus = RandomUnderSampler(random_state=42)

datas_unders, labels_unders = rus.fit_sample(datas, labels)

print('Undersampled dataset shape {}'.format(Counter(labels_unders)))


# %%
# plotting scatter distributions
#


color = []
X = []
Y = []
feature1 = mad
feature2 = quality

for a, x, y in zip(labels, feature1, feature2):
  if(a==1):
    color.append('r')
  else:
    color.append('b')

  X.append(x)
  Y.append(y)

figure, ax = plt.subplots()
ax.set_yscale("log")
ax.set_ylim(1e-7, 1)
ax.scatter(X, Y, c=color, s=1)


# %%
# plotting histograms distributions


X1 = []
X0 = []
feature = mad
label = age

for lbl, f in zip(label, feature):
  upper_limit = 90
  lower_limit = 60
  l = int(lbl)
  if(lower_limit <= l <= upper_limit):
    X0.append(f)

#X1 = np.asarray(X1)/len(X1)
#X0 = np.asarray(X0)/len(X0)
fig, ax = plt.subplots(1, 1, sharex=True)

#bins = np.linspace(min(feature), max(feature), 100)
bins = np.linspace(0, 120, 200)

ax.hist(X0, bins=bins, label=str(lower_limit)+" < age < "+str(upper_limit))
ax.set_title("tpr")
ax.set_ylim(0,100)
ax.legend()


# %%
# train-test splitting


#Splitting into a training set and a test set using a stratified k fold
datas_train, datas_test, labels_train, labels_test = train_test_split(
    datas_unders, labels_unders, test_size=0.25, random_state=42)
print('train {}'.format(Counter(labels_train)))
print('test {}'.format(Counter(labels_test)))


# %%
# PCA


# plotting explained variance
Pca = PCA().fit(datas_unders)
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

ax1.set_title("PCA")
ax1.plot(np.cumsum(Pca.explained_variance_ratio_))
ax1.set_ylabel('cumulative explained variance')

ax2.plot(Pca.explained_variance_ratio_)

ax2.set_xlabel('number of components')
ax2.set_ylabel('explained variance')

ax1.grid(True)
ax2.grid(True)
plt.show()


# making PCA
number_of_components = 2

# pca for train-test splitted datas
pca_tt = PCA(n_components=number_of_components, svd_solver='randomized',
             whiten=True).fit(datas_train)
datas_train_pca = pca_tt.transform(datas_train)
datas_test_pca = pca_tt.transform(datas_test)

# pca for whole datas
pca = PCA(n_components=number_of_components, svd_solver='randomized',
          whiten=True).fit(datas_unders)
datas_pca = pca.transform(datas_unders)


# plotting 2 pca components
component_1 = list(map(itemgetter(0), datas_pca))
component_2 = list(map(itemgetter(1), datas_pca))
co = []
compx = []
compy = []

for l in labels_train:
    if(l==0.):
        co.append('r')
    else:
        co.append('b')

fig, ax = plt.subplots()
ax.scatter(component_1, component_2, s=1, c=co)


# %%
# Unsupervised learning


predicted_labels = skc.SpectralClustering(random_state=4,
                                          n_clusters=2).fit_predict(datas_unders)

# labels inversion (in case we need to exchange zeros with ones and viceversa)
#predicted_labels = abs(np.asarray(predicted_labels) - 1)

print(Counter(predicted_labels))
print(classification_report(labels_unders, predicted_labels))


# %%
# Supervised learning


# Train a SVM classification model
print("Fitting the classifier to the training set")
param_grid = {'C': [1e4, 6e4, 1e5, 5e5, 9e2, 1e6, 4e3],
              'gamma': [2e-5,1e-5,5e-6,1e-6,5e-7], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=3,
                   verbose=3, return_train_score=True)

clf = clf.fit(datas_train_pca, labels_train)

print("Best estimator found by grid search:")
print(clf.best_estimator_)
print("more infos:")
print(clf.cv_results_)
print("more infos:")
print(clf.best_score_)
print("more infos:")
print(clf.scorer_)
print("more infos:")
print(clf.best_index_)


# Quantitative evaluation of the model quality on the test set
print("Predicting labels on the test set")
predicted_lbl = clf.predict(datas_pca)

print(classification_report(labels_unders, predicted_lbl))
print(confusion_matrix(labels_unders, predicted_lbl, labels=range(2)))


# %%
# TPOT


tpot = TPOTClassifier(generations=100, population_size=100, max_time_mins=1000, n_jobs=-1, random_state=42, early_stop=5, verbosity=3)
#tpot.fit(datas_train_pca, labels_train)  # if you want to use pca first
tpot.fit(datas_train, labels_train)  # if you don't want to use pca first
tpot.export('tpot_pipeline.py')
print("SCORE:")
print(tpot.score(datas_test, labels_test))
