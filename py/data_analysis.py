
# -*- coding: utf-8 -*-

# cardio stuff
import clean_db

# standard libraries
import numpy as np
import matplotlib.pylab as plt
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import RidgeCV, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Load already cleaned db and convert to dataframe
f = "../cardio_final.json"
df = clean_db.db_to_dataframe(f)

# %%

# Clean file for further use
df = clean_db.clean_db(df,
                       drop_columns=['rhythm', 'city', 'country',
                                     'filename', 'filenumber'],
                       drop_na_in=['weight', 'tpr', 'madRR', 'medianRR',
                                   'opt_delay', 'afib', 'age', 'sex', 'smoke',
                                   'afib', 'bmi', 'lifestyle'],
                       drop_zeroes_in=['weight', 'age', 'length'],
                       quality_threshold=None, reset_index=True)

# CONVERTING STRING LABELS TO ARBITRARY NUMERICAL VALUES
df = df.replace('F', -1).replace('M', 1)
df = df.replace('C1', 1).replace('C0', 0).replace('C3', 3).replace('C2', 2)
df = df.replace('Active', 3
                ).replace('Actief', 3
                ).replace('Gemiddeld', 2
                ).replace('Moderate', 2
                ).replace('Sedentary', 1
                ).replace('Weinig', 1)


# remove array features (not cleaned before because they were used for double
# gaussian fitting)
df = df.drop(columns=['RR', 'AA', 'Rpeakvalues', 'time', 'signal'])
df = df.astype(float)

# %%
# outlier removal

# list of features where to perform outlier removal
s = ['weight', 'length', 'bmi', 'lf', 'hf', 'c', 'bpm', 'sdsd', 'rmssd',
     'entropyRR', 'opt_delay', 'pnn20', 'medianRR', 'medianAA', 'skewnessRR',
     'kurtosisRR', 'skewnessAA', 'madRR', 'madAA', 'meanAA', 'diff_mad',
     'dist_mad', 'n_diff_mad', 'n_dist_mad', 'diff_mad_all', 'dist_mad_all',
     'n_dist_mad_all', 'n_diff_mad_all', 'diff_std', 'dist_std', 'n_diff_std',
     'n_dist_std', 'diff_std_all', 'dist_std_all', 'n_dist_std_all',
     'n_diff_std_all']

# NOTE: outlier removal performed over every feature basically empties the db
# highlighting really sparse data, we decided which feature to use trying to
# keep at least 1000 patients and raising as much as possible the score


# a copy of the db is needed if we want to execute more than once this cell
# without reloading the json
d = df
outlier_free = d

# number of mad kept on each tail from the median of the distribution
Nmad = 4.
for _ in s:
  # mean and mad computed from the starting distributions
  m = np.median(d[_])
  mad = np.median(np.abs(d[_] - m))
  outlier_free = outlier_free[np.logical_and(outlier_free[_] > m - Nmad*mad,
                                             outlier_free[_] < m + Nmad*mad)]
d = outlier_free

# %%

# choose which threshold to apply to further analysis data
# IMPORTANT: quality is a bad quality index, so the higher the worse
bad_quality_threshold = .01
d = d[d.quality < bad_quality_threshold]

# %%

# if you prefer you can set a threshold on the correlation needed to select a
# feature for the downcoming analysis. we preferred to use all the features
correlation_threshold = 0.0

c = d.corr(method='pearson')
Features = c[abs(c.age) > correlation_threshold]
x = d[Features.index.values]

# from now on, x is the db containing the data and target is the series
# containing patients' ages
target = x.age
x = x.drop(columns='age')

# %%

# Scale before PCA
stdsc = StandardScaler()
standardized_x = stdsc.fit_transform(x)

# %%

# plotting explained variance and setting number of components to use by
# choosing a cumulative explained variance threshold
Pca = PCA().fit(standardized_x)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axis_x = np.arange(1, Pca.n_components_+1)

ty = 0.99
tx = np.where(np.cumsum(Pca.explained_variance_ratio_) > ty)[0][1]

# visualizing the thersholds and the explained variance
ax1.set_title("PCA")
ax1.plot(axis_x, 100*np.cumsum(Pca.explained_variance_ratio_))
ax1.plot([tx, tx], [20, 100], 'r-.')
ax1.plot([0, Pca.n_components_], [100*ty, 100*ty], 'k-.',
         label="explained variance threshold = "+str(ty*100)+"%")

ax2.plot(axis_x, 100*Pca.explained_variance_ratio_)
ax2.plot([tx, tx], [-1, 100*Pca.explained_variance_ratio_[0]], 'r-.',
         label='component threshold = '+str(tx))

ax1.set_ylabel('cumulative \n explained variance (%)')
ax2.set_xlabel('number of components', fontsize=14)
ax2.set_ylabel('explained variance (%)', fontsize=14)

ax1.set_xlim(1,  Pca.n_components_+1)

ax1.grid(True)
ax2.grid(True)

ax1.legend(loc=3)
ax2.legend()

plt.show()

# %%

Pca_with_n_components = PCA(n_components=tx)
pca_x = Pca_with_n_components.fit_transform(standardized_x)

a = np.linspace(100, 300, 200)  # to tune properly if you change anything above
ridge = RidgeCV(alphas=a, cv=10)
ridge.fit(pca_x, target)

predicted = ridge.predict(pca_x)

# computing some statistical measures
stat_pearson = pearsonr(predicted, target)
stat_spearman = spearmanr(predicted, target)
stat_kendall = kendalltau(predicted, target)

print("ridge alpha: ", ridge.alpha_)
print("corr: ", stat_pearson)
print("rho: ", stat_spearman)
print("tau: ", stat_kendall)
print("score: ", ridge.score(pca_x, target))

# needed to work properly
predicted = predicted.reshape(-1, 1)

# A linear regression is needed to visually compare predicted age vs real age
LR = LinearRegression().fit(predicted, target)
predicted = LR.predict(predicted)

# needed to work properly
predicted = predicted.reshape(-1, 1)

# plot
plt.scatter(target, predicted, s=3, color="black")
plt.title("Regression after PCA")
plt.plot([min(target), max(target)],
         [min(target)*LR.coef_[0]+LR.intercept_,
          max(target)*LR.coef_[0]+LR.intercept_],
         'r-.',
         label="Regression line Y = " + str(LR.coef_[0])[:5] + "*X + " +
               str(LR.intercept_)[:6])

plt.xlabel("Real age (years)")
plt.ylabel("Predicted age (years)")

plt.grid()
plt.legend()

# %%

kf = KFold(n_splits=10, shuffle=True, random_state=1000000000)
final_alpha = ridge.alpha_  # fixing alpha

# container for pearson r, spearman rho, kendall tau and score for each fold
corrt, scoret, rho, tau = [], [], [], []

for train_index, test_index in kf.split(pca_x, target):
  # training
  new_ridge = Ridge(alpha=final_alpha)
  new_ridge.fit(pca_x[train_index], target.iloc[train_index])
  new_predicted = new_ridge.predict(pca_x[test_index])

  # measuring performances
  corrt.append(pearsonr(new_predicted, target.iloc[test_index]))
  rho.append(spearmanr(new_predicted, target.iloc[test_index]))
  tau.append(kendalltau(new_predicted, target.iloc[test_index]))
  scoret.append(new_ridge.score(pca_x[test_index], target.iloc[test_index]))

# mean values over the 10-fold
corrt_ = np.mean(corrt, axis=0)[0]
scoret_ = np.mean(scoret)
rho_ = np.mean(rho, axis=0)[0]
tau_ = np.mean(tau, axis=0)[0]


# printing reults of mean and max for each list
print("   measure      mean        max    (p-value of max)")
print("R^2:          %2.5f,    %2.5f" % (scoret_, max(scoret)))
print("Pearson R:    %2.5f,    %2.5f     %2.7f" % (corrt_, max(corrt)[0], max(corrt)[1]))
print("Spearman Rho: %2.5f,    %2.5f     %2.7f" % (rho_, max(rho)[0], max(rho)[1]))
print("Kendall Tau:  %2.5f,    %2.5f     %2.7f" % (tau_, max(tau)[0], max(tau)[1]))

# OLD RESULTS
# Pearson r
#      mean                         (max, p-value of max)
# 0.6871186240192718 (0.7774754922030191, 6.07521505515027e-27)
# score
#      mean                  max
# 0.46468027461675787 0.59137684046488


# CURRENT RESULTS
# Pearson r
#      mean                         (max, p-value of max)
# 0.7452870656680132 (0.8406409487826519, 1.0219843095920135e-40)
# score
#      mean                  max
# 0.5490146030858843 0.6890449306017845
