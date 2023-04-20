# -*- coding: utf-8 -*-
"""
Optimal transport in synthetic data with type1 or type2 outliers
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

# parameter
outlier_list = ['None', 'type1', 'type2', 'both']
outlier_flag = outlier_list[1]
plot_flag = 0

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

elif outlier_flag == 'both':
    Xs_o, ys_o = ot.datasets.make_data_classif('gaussrot', 30, nz=noise_level, random_state=119)
    Xs_o = Xs_o - 3
    Xs_o[ys_o == 1] += [2, -2]
    Xs_o[ys_o == 2] += [-2, +2]
    Xs = np.vstack((Xs, Xs_o))
    ys = np.append(ys, ys_o)
    Xs_o, ys_o = ot.datasets.make_data_classif('gaussrot', 30, nz=noise_level, random_state=119)
    Xs_o = Xs_o + 3
    Xs_o[ys_o == 1] += [3,-2]
    Xs_o[ys_o == 2] += [-2,1]
    Xs = np.vstack((Xs, Xs_o))
    ys = np.append(ys, ys_o)

else:
    print("None")

if plot_flag == 1:
    plot_data(Xs, ys, Xt, yt)

Accs = []
#### OT ####
reg_e = 1
ot_sinkhorn = ot.da.SinkhornTransport(reg_e, max_iter=10000)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
Gs = ot_sinkhorn.coupling_
if plot_flag == 1:
    plot_transp(Xs, ys, Xt, yt, Gs, reg_e)
# mapping
transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
if plot_flag == 1:
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
if plot_flag == 1:
    plot_transp(Xs, ys, Xt, yt, Gs, reg_e, reg_m, method='UOT')
# mapping
transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
if plot_flag == 1:
    plot_mapping(transp_Xs_sinkhorn, ys, Xt, yt, method='UOT')
# kNN
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
y_pred = knc.predict(Xt)
acc = knc.score(Xt, yt)
# print(f"Accuracy: {score*100}%")
Accs.append(acc*100)

print(Accs)