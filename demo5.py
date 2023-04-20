# -*- coding: utf-8 -*-
"""
Optimal Transport for Domain Adaptation with synthetic data including outliers and SVM experiments
"""
# Author: T.Tabuchi
# Date  : 2022/1/18

from utils import plot_data, plot_transp, plot_mapping

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import ot
import ot.plot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# set up
outlier_list = ['None', 'type1', 'type2']
outlier_flag = outlier_list[1]

# Generate synthetic data
n_source_samples = 100
n_target_samples = 100
theta = 2 * np.pi / 20
noise_level = 0.1

Xs, ys = ot.datasets.make_data_classif('gaussrot', n_source_samples, nz=noise_level, random_state=111)
Xt, yt = ot.datasets.make_data_classif('gaussrot', n_target_samples, theta=theta, nz=noise_level, random_state=112)

# one of the target mode changes its variance (no linear mapping)
Xt[yt == 2] *= 3
Xt = Xt + 4

if outlier_flag == 'type1':
    Xs_o, ys_o = ot.datasets.make_data_classif('gaussrot', 30, nz=noise_level, random_state=119)
    Xs_o = Xs_o - 3
    Xs_o[ys_o == 1] += [2, -2]
    Xs_o[ys_o == 2] += [-2, +2]
    Xs = np.vstack((Xs, Xs_o))
    ys = np.append(ys, ys_o)

elif outlier_flag == 'type2':
    Xs_o, ys_o = ot.datasets.make_data_classif('gaussrot', 30, nz=noise_level, random_state=119)
    Xs_o = Xs_o + 3
    Xs_o[ys_o == 1] += [3,-2]
    Xs_o[ys_o == 2] += [-2,1]
    Xs = np.vstack((Xs, Xs_o))
    ys = np.append(ys, ys_o)
else:
    print("None")

# SVM Classification
X = np.vstack((Xs, Xt))
y = np.hstack((np.zeros(Xs.shape[0]), np.ones(Xt.shape[0])))

clf = SVC(kernel='linear')
clf.fit(X, y)
coef = clf.coef_[0]
intercept = clf.intercept_

pl.figure()
line = np.linspace(-2, 10)
pl.plot(line, -(line * coef[0] + intercept) / coef[1], c='g', label="SVM (linear)")
pl.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=cm.coolwarm, marker='+', label='Source samples')
pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap=cm.coolwarm, marker='o', label='Target samples')
pl.legend(loc=0)
pl.title('Source and target distributions')
pl.show()

ind1, = np.where(clf.predict(Xs)!=0)
ind2, = np.where(clf.predict(Xt)!=1)
print(ind1.shape)
print(ind2.shape)

Xs = np.delete(Xs, ind1, 0)
Xt = np.delete(Xt, ind2, 0)
ys = np.delete(ys, ind1, 0)
yt = np.delete(yt, ind2, 0)


Accs = []
#### kNN (no adaptation) ####
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(Xs, ys)
acc = knc.score(Xt, yt)
Accs.append(acc*100)


#### OT ####
reg_e = 1
ot_sinkhorn = ot.da.SinkhornTransport(reg_e, max_iter=10000)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
Gs = ot_sinkhorn.coupling_
plot_transp(Xs, ys, Xt, yt, Gs, reg_e)
# mapping
transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
plot_mapping(transp_Xs_sinkhorn, ys, Xt, yt)
# kNN
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
y_pred = knc.predict(Xt)
acc = knc.score(Xt, yt)
# print(f"Accuracy: {score*100}%")
Accs.append(acc*100)


#### UOT ####
reg_e = 1
reg_m = 10
ot_sinkhorn = ot.da.UnbalancedSinkhornTransport(reg_e, reg_m, max_iter=10000)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
Gs = ot_sinkhorn.coupling_
plot_transp(Xs, ys, Xt, yt, Gs, reg_e, reg_m, method='UOT')
# mapping
transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
plot_mapping(transp_Xs_sinkhorn, ys, Xt, yt, method='UOT')
# kNN
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
y_pred = knc.predict(Xt)
acc = knc.score(Xt, yt)
# print(f"Accuracy: {score*100}%")
Accs.append(acc*100)

print(Accs)