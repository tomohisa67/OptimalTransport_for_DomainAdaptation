# -*- coding: utf-8 -*-
"""
Optimal Transport for Domain Adaptation with Real Data
"""
# Author: T.Tabuchi
# Date  : 2022/1/11

from utils import load_data, class_balancing
from utils import transp_source_to_target, create_Ls, search_neighbor

import numpy as np
import scipy.io as sio
import ot
import ot.plot
from sklearn.neighbors import KNeighborsClassifier
import random
import csv


domains = ["caltech", "amazon", "webcam", "dslr"]
algs = ["1NN", "OT", "OTc", "UOT", "UOTc", "UOTn"]

# parameters
balancing = 0  # balance class ratios or not
addoutlier = 0 # add outliers or not
numoutd = 3    # the number of type 2 outliers
csvfile = 1    # create csv file
filename = "result.csv"

reg_e = 0.1
reg_m = 0.1
p = 1

Accs_list = []
for alg in algs:
    Tasks = []
    Accs = []
    for source in domains:
        for target in domains:
            # print("source:{} --> target:{}".format(source, target))

            source_data = sio.loadmat("./data/surf/{}_surf_10.mat".format(source))
            target_data = sio.loadmat("./data/surf/{}_surf_10.mat".format(target))
            Xs = source_data["feas"]
            ys = source_data["label"].reshape(1,-1)[0]
            Xt = target_data["feas"]
            yt = target_data["label"].reshape(1,-1)[0]

            # balance class ratios
            if balancing == 1:
                u, counts = np.unique(ys, return_counts=True)
                num = min(counts)
                Xs, ys = class_balancing(Xs, ys, num=num, seed=1)

                u, counts = np.unique(yt, return_counts=True)
                num = min(counts)
                Xt, yt = class_balancing(Xt, yt, num=num, seed=1)

            # add type2 outliers
            if addoutlier == 1:
                xx_list = []
                yy_list = []
                classes = np.unique(ys)
                for c in classes:
                    idxc, = np.where(ys == c)
                    lsize = idxc.shape[0]
                    Gd = np.sum(Xs[idxc], axis=0) / lsize

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

                Xs = np.concatenate([Xs, np.array(xx_list)])
                ys = np.concatenate([ys, np.array(yy_list)])

            ns = Xs.shape[0]
            nt = Xt.shape[0]

            a = np.ones(ns) / ns
            b = np.ones(nt) / nt
            M = ot.dist(Xs, Xt, metric='sqeuclidean')
            M = M / M.max()

            if alg == "1NN":
                knc =  KNeighborsClassifier(n_neighbors=1)
                knc.fit(Xs, ys)
                acc = knc.score(Xt, yt)

            if alg == "OT":
                # reg_e = 10 # entropy
                Gs = ot.sinkhorn(a, b, M, reg_e, method='sinkhorn', numItermax=100000)
                transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
                knc =  KNeighborsClassifier(n_neighbors=1)
                knc.fit(transp_Xs_sinkhorn, ys)
                acc = knc.score(Xt, yt)
            
            elif alg == "OTc":
                # p = 1
                # reg_e = 10 # entropy
                Ls = create_Ls(Xs, ys)
                Ls = Ls / Ls.max()
                M1 = M + p * Ls
                M1 = M1 / M1.max()
                Gs = ot.sinkhorn(a, b, M1, reg_e, method='sinkhorn', numItermax=100000)
                transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
                knc =  KNeighborsClassifier(n_neighbors=1)
                knc.fit(transp_Xs_sinkhorn, ys)
                acc = knc.score(Xt, yt)

            elif alg == "UOT":
                # reg_e = 10 # entropy
                # reg_m = 10 # KL
                Gs = ot.sinkhorn_unbalanced(a, b, M, reg_e, reg_m, method='sinkhorn', numItermax=100000)
                transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
                knc =  KNeighborsClassifier(n_neighbors=1)
                knc.fit(transp_Xs_sinkhorn, ys)
                acc = knc.score(Xt, yt)
            
            elif alg == "UOTc":
                # p = 1
                # reg_e = 10 # entropy
                # reg_m = 10 # KL
                Ls = create_Ls(Xs, ys)
                Ls = Ls / Ls.max()
                M1 = M + p * Ls
                M1 = M1 / M1.max()
                Gs = ot.sinkhorn_unbalanced(a, b, M1, reg_e, reg_m, method='sinkhorn', numItermax=100000)
                transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
                knc =  KNeighborsClassifier(n_neighbors=1)
                knc.fit(transp_Xs_sinkhorn, ys)
                acc = knc.score(Xt, yt)

            elif alg == "UOTn":
                # p = 1
                # reg_e = 10 # entropy
                # reg_m = 10 # KL
                Ln = search_neighbor(Xs, ys)
                Ln = Ln / Ln.max()
                Ls = create_Ls(Xs, ys)
                Ls = Ls / Ls.max()
                M1 = M + p * Ls + p * 10 * Ln
                M1 = M1 / M1.max()
                Gs = ot.sinkhorn_unbalanced(a, b, M1, reg_e, reg_m, method='sinkhorn', numItermax=100000)
                transp_Xs_sinkhorn = transp_source_to_target(Xt, Gs)
                knc =  KNeighborsClassifier(n_neighbors=1)
                knc.fit(transp_Xs_sinkhorn, ys)
                acc = knc.score(Xt, yt)
            

            Tasks.append(source[0].upper()+"->"+target[0].upper())
            Accs.append(round(acc*100,2))
    Tasks.append("avg")
    Accs.append(round(np.mean(np.array(Accs)),2))
    Accs_list.append(Accs)
    # print("task:\tacc")
    # for k in range(len(Tasks)):
    #     print("{:}:\t{:.2f}".format(Tasks[k],Accs[k]))


Tasks.insert(0, '')
if csvfile == True:
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(Tasks)
        for i, row in zip(algs, Accs_list):
            writer.writerow([i] + row)

    import pandas as pd
    df = pd.read_csv(filename)
    df = df[['Unnamed: 0', 'C->A', 'C->W', 'C->D', 
                'A->C', 'A->W', 'A->D', 
                'W->C', 'W->A', 'W->D', 
                'D->C', 'D->A', 'D->W', 'avg']]
    df = df.set_index('Unnamed: 0')
    df = df.rename(index={'avg': 'means'})
    df = df.T
    df.to_csv(filename)