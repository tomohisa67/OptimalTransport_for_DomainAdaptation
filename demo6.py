# -*- coding: utf-8 -*-
"""
Optimal transport in synthetic data with type1 or type2 outliers (Experiments of the proposed method)
"""
# Author: T.Tabuchi
# Date  : 2022/1/19

from utils import plot_data, plot_transp, plot_mapping
from utils import transp_source_to_target, create_Ls

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import ot
import ot.plot
from sklearn.neighbors import KNeighborsClassifier


# parameter
outlier_list = ['None', 'type1', 'type2', 'both']
outlier_flag = outlier_list[2]
plot_flag = 1
p = 5

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


ns = Xs.shape[0]
nt = Xt.shape[0]

a = np.ones(ns) / ns
b = np.ones(nt) / nt
M = ot.dist(Xs, Xt, metric='sqeuclidean')
M = M / M.max()


Accs = []
#### kNN (no adaptation) ####
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(Xs, ys)
acc = knc.score(Xt, yt)
Accs.append(acc)


#### OT ####
reg_e = 0.01 # entropy
Gs = ot.sinkhorn(a, b, M, reg_e, method='sinkhorn', numItermax=100000)
if plot_flag == 1:
    plot_transp(Xs, ys, Xt, yt, Gs, reg_e, method='OT')
transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
if plot_flag == 1:
    plot_mapping(transp_Xs_sinkhorn, ys, Xt, yt, method='OT')
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
acc = knc.score(Xt, yt)
Accs.append(acc)


#### UOT ####
reg_e = 0.01 # entropy
reg_m = 0.1 # KL
Gs = ot.sinkhorn_unbalanced(a, b, M, reg_e, reg_m, method='sinkhorn', numItermax=100000)
if plot_flag == 1:
    plot_transp(Xs, ys, Xt, yt, Gs, reg_e, reg_m, method='UOT')
transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
if plot_flag == 1:
    plot_mapping(transp_Xs_sinkhorn, ys, Xt, yt, method='UOT')
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
acc = knc.score(Xt, yt)
Accs.append(acc)


M = ot.dist(Xs, Xt, metric='sqeuclidean')
M = M / M.max()
#### OTc ####
# p = 1
reg_e = 0.01 # entropy
Ls = create_Ls(Xs, ys)
Ls = Ls / Ls.max()
M = M + p * Ls
M = M / M.max()
Gs = ot.sinkhorn(a, b, M, reg_e, method='sinkhorn', numItermax=100000)
if plot_flag == 1:
    plot_transp(Xs, ys, Xt, yt, Gs, reg_e, method='OT')
transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
if plot_flag == 1:
    plot_mapping(transp_Xs_sinkhorn, ys, Xt, yt, method='OT')
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
acc = knc.score(Xt, yt)
Accs.append(acc)


M = ot.dist(Xs, Xt, metric='sqeuclidean')
M = M / M.max()
#### UOTc ####
# p = 1
reg_e = 0.01 # entropy
reg_m = 0.1 # KL
Ls = create_Ls(Xs, ys)
Ls = Ls / Ls.max()
M = M + p * Ls
M = M / M.max()
Gs = ot.sinkhorn_unbalanced(a, b, M, reg_e, reg_m, method='sinkhorn', numItermax=100000)
if plot_flag == 1:
    plot_transp(Xs, ys, Xt, yt, Gs, reg_e, reg_m, method='UOT')
transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
if plot_flag == 1:
    plot_mapping(transp_Xs_sinkhorn, ys, Xt, yt, method='UOT')
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
acc = knc.score(Xt, yt)
Accs.append(acc)


print(Accs)