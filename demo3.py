# -*- coding: utf-8 -*-
"""
Optimal Transport for Domain Adaptation with Synthetic Data (simple version)
"""
# Author: T.Tabuchi
# Date  : 2022/1/11

from utils import load_data, class_balancing
from utils import transp_source_to_target, create_Ls

import numpy as np
import scipy.io as sio
import ot
import ot.plot
from sklearn.neighbors import KNeighborsClassifier
import random


#### Load data ####
source = "dslr" # ["caltech", "amazon", "webcam", "dslr"]
target = "webcam"

source_data = sio.loadmat("./data/surf/{}_surf_10.mat".format(source))
target_data = sio.loadmat("./data/surf/{}_surf_10.mat".format(target))

Xs = source_data["feas"]
ys = source_data["label"].reshape(1,-1)[0]

Xt = target_data["feas"]
yt = target_data["label"].reshape(1,-1)[0]

balancing = 0
if balancing == 1:
    u, counts = np.unique(ys, return_counts=True)
    num = min(counts)
    Xs, ys = class_balancing(Xs, ys, num=num, seed=1)

    u, counts = np.unique(yt, return_counts=True)
    num = min(counts)
    Xt, yt = class_balancing(Xt, yt, num=num, seed=1)
    # numoutw = 6

ns = Xs.shape[0]
nt = Xt.shape[0]

numoutd = 3
addoutlier = 1
if addoutlier == 1:
    # add type2 outliers
    xx_list = []
    yy_list = []
    classes = np.unique(ys)
    for c in classes:
        idxc, = np.where(ys == c)
        lsize = idxc.shape[0]
        Gd = np.sum(Xs[idxc], axis=0) / lsize

        # random.seed(111)
        # L = range(10) + 1
        # L.pop(c)
        # j = random.randint(L)

        # idxc, = np.where(yt == j)
        # lsize = idxc.shape[0]
        # Gw = np.sum(Xt[idxc], axis=0) / lsize

        if c != 10:
            idxc, = np.where(yt == c+1)
            lsize = idxc.shape[0]
            Gw = np.sum(Xt[idxc], axis=0) / lsize
        else:
            idxc, = np.where(yt == 1)
            lsize = idxc.shape[0]
            Gw = np.sum(Xt[idxc], axis=0) / lsize

        # G = (Gd + 9 * Gw) / 10
        G = Gw
        random.seed(111)
        noise = [random.random() - 0.5 for i in range(800)]
        noise = np.array(noise) * 10
        for i in range(numoutd):
            xx = G + noise
            yy = c
            xx_list.append(xx)
            yy_list.append(yy)

        # xx = G
        # yy = c
        # for i in range(numoutd):
        #     xx_list.append(xx)
        #     yy_list.append(yy)
    
    xx_arr = np.array(xx_list)
    yy_arr = np.array(yy_list)

    Xs = np.concatenate([Xs, xx_arr])
    ys = np.concatenate([ys, yy_arr])

#### kNN (no adaptation) ####
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(Xs, ys)
acc = knc.score(Xt, yt)
print(f"kNN Accuracy: {acc*100}%")

#### OT ####
reg_e = 1 # entropy # outlierなし: 0.03
ns = Xs.shape[0]
nt = Xt.shape[0]
a = np.ones(ns) / ns
b = np.ones(nt) / nt
M = ot.dist(Xs, Xt, metric='sqeuclidean')
M = M / M.max()
Gs = ot.sinkhorn(a, b, M, reg_e, method='sinkhorn', numItermax=100000)
transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
acc = knc.score(Xt, yt)
print(f"OT Accuracy: {acc*100}%")

#### UOT ####
reg_e = 1 # entropy # outlierなし: 0.03
reg_m = 10 # KL # outlierなし: 0.08
ns = Xs.shape[0]
nt = Xt.shape[0]
a = np.ones(ns) / ns
b = np.ones(nt) / nt
M = ot.dist(Xs, Xt, metric='sqeuclidean')
M = M / M.max()
Gs = ot.sinkhorn_unbalanced(a, b, M, reg_e, reg_m, method='sinkhorn', numItermax=100000)
transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
acc = knc.score(Xt, yt)
print(f"UOT Accuracy: {acc*100}%")

#### UOTc ####
p = 100
reg_e = 1 # entropy # outlierなし: 0.003
reg_m = 10 # KL # outlierなし: 0.01
ns = Xs.shape[0]
nt = Xt.shape[0]
a = np.ones(ns) / ns
b = np.ones(nt) / nt
M = ot.dist(Xs, Xt, metric='sqeuclidean')
Ls = create_Ls(Xs, ys)
M = M + p * Ls
M = M / M.max()
Gs = ot.sinkhorn_unbalanced(a, b, M, reg_e, reg_m)
transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(transp_Xs_sinkhorn, ys)
acc = knc.score(Xt, yt)
print(f"UOTc Accuracy: {acc*100}%")