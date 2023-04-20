# -*- coding: utf-8 -*-
"""
Optimal Transport for Domain Adaptation with Real Data (simple version)
"""
# Author: T.Tabuchi
# Date  : 2022/1/11

from utils import load_data, class_balancing

import numpy as np
import matplotlib.pyplot as pl
import ot
import ot.plot
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier

#### Load data ####
source = "dslr" # ["caltech", "amazon", "webcam", "dslr"]
target = "webcam"

source_data = sio.loadmat("./data/surf/{}_surf_10.mat".format(source))
target_data = sio.loadmat("./data/surf/{}_surf_10.mat".format(target))

Xs = source_data["feas"]
ys = source_data["label"].reshape(1,-1)[0]

Xt = target_data["feas"]
yt = target_data["label"].reshape(1,-1)[0]

ns = Xs.shape[0]
nt = Xt.shape[0]

#### OTDA ####
reg_e = 100
ot_sinkhorn = ot.da.SinkhornTransport(reg_e)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
Gs = ot_sinkhorn.coupling_

transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)

X_train = transp_Xs_sinkhorn
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(X_train, ys)
y_pred = knc.predict(Xt)
score = knc.score(Xt, yt)
print(f"OTDA Accuracy: {score*100}%")

#### UOTDA ####
reg_e = 100
reg_m = 1000
uot_sinkhorn = ot.da.UnbalancedSinkhornTransport(reg_e, reg_m)
uot_sinkhorn.fit(Xs=Xs, ys=ys, Xt=Xt)
Gs = uot_sinkhorn.coupling_

transp_Xs_uot = uot_sinkhorn.transform(Xs=Xs)

X_train = transp_Xs_uot
knc =  KNeighborsClassifier(n_neighbors=1)
knc.fit(X_train, ys)
y_pred = knc.predict(Xt)
score = knc.score(Xt, yt)
print(f"UOTDA Accuracy: {score*100}%")