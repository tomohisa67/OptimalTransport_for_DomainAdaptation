# -*- coding: utf-8 -*-
"""
Various useful functions
"""
# Author: T.Tabuchi
# Date  : 2022/1/11

import numpy as np
import scipy.io as sio
from scipy.spatial import distance
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import ot
import ot.plot


def load_data(source, target, num_labeled=1, class_balance=True, num=10, seed=0):
    source_data = sio.loadmat("./data/surf/{}_surf_10.mat".format(source))
    target_data = sio.loadmat("./data/surf/{}_surf_10.mat".format(target))

    source_feat, source_label = source_data["fts"], source_data["labels"]
    target_feat, target_label = target_data["fts"], target_data["labels"]
    source_label, target_label = source_label.reshape(-1, 1), target_label.reshape(-1, 1)

    indexes = sio.loadmat("./data/labeled_index/{}_{}.mat".format(target,num_labeled))
    idx_labeled, idx_unlabeled = indexes["labeled_index"][0], indexes["unlabeled_index"][0]

    target_feat_l, target_label_l = target_feat[idx_labeled], target_label[idx_labeled]
    target_feat_un, target_label_un = target_feat[idx_unlabeled], target_label[idx_unlabeled]

    if class_balance:
        source_feat, source_label = class_balancing(source_feat, source_label, num=num,seed=seed)
        target_feat_un, target_label_un = class_balancing(target_feat_un, target_label_un, num=num)

    return source_feat, source_label, target_feat_l, target_label_l, target_feat_un, target_label_un


def class_balancing(x, t, num=10, seed=0):
    t = np.squeeze(t)
    x_class = []
    n_class = t.max()
    for k in range(1, n_class + 1, 1):
        x_class.append(x[t == k])
    np.random.seed(seed)
    x_class_balance = [xx[np.random.choice(np.arange(len(xx)), num, replace=True)] for xx in x_class]
    t_class_balance = [np.ones(num, dtype=np.int32) * k for k in range(1, n_class + 1, 1)]
    x_class_balance = np.vstack(x_class_balance)
    t_class_balance = np.hstack(t_class_balance)
    return x_class_balance, t_class_balance


def create_Ls(Xs, ys):
    ns = Xs.shape[0]
    Ds = np.zeros(ns)

    indices_labels = []
    classes = np.unique(ys)
    for c in classes:
        idxc, = np.where(ys == c)
        indices_labels.append(idxc)

        X = Xs[idxc]
        n = X.shape[0]
        D = []
        X_sum = np.sum(X, axis=0) 

        for k in range(n):
            G = (X_sum - X[k]) / (n - 1)
            d = np.linalg.norm(G - X[k], ord=1)
            D = np.append(D, d)
        Ds[idxc] = D

    Ls = Ds.reshape(-1,1)
    return Ls


def search_neighbor(Xs, ys):
    ns = Xs.shape[0]
    Ds = np.zeros(ns)

    classes = np.unique(ys)
    for c in classes:
        idxc, = np.where(ys == c)
        dist_X =  distance.cdist(Xs[idxc], Xs[idxc], metric='euclidean')
        dist_X = np.sort(dist_X)
        dist_X = dist_X[:,:3]
        dist_X = np.sum(dist_X, axis=1) / 3
        Ds[idxc] = dist_X

    Ln = Ds.reshape(-1,1)
    return Ln 


def transp_source_to_target(Xt, T):
    nt = T.shape[1]
    transp_Xs = np.diag(1 / (T @ np.ones(nt))) @ T @ Xt
    return transp_Xs


def plot_data(Xs, ys, Xt, yt):
    pl.figure()
    pl.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=cm.coolwarm, marker='+', label='Source samples')
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap=cm.coolwarm, marker='o', label='Target samples')
    pl.legend(loc=0)
    pl.title('Source and target distributions')
    pl.show()


def plot_transp(Xs, ys, Xt, yt, Gs, reg_e, reg_m=10, method='OT', savefig=0, path='~/Desktop/', filename='fig'):
    pl.figure()
    ot.plot.plot2D_samples_mat(Xs, Xt, Gs, zorder=1, alpha=0.5)
    pl.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=cm.coolwarm, marker='+', label='Source samples', zorder=2)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap=cm.coolwarm, marker='o', label='Target samples', zorder=2)
    pl.legend(loc=0)
    if method == 'UOT':
        pl.title(f'OT matrix Sinkhorn with samples: reg_e={reg_e}, reg_m={reg_m}')
    else:
        pl.title(f'OT matrix Sinkhorn with samples: reg_e={reg_e}')
    if savefig == 1:
        pl.savefig(path + filename)
    # pl.show()


def plot_mapping(transp_Xs, ys, Xt, yt, method='OT', savefig=0, path='~/Desktop/', filename='fig'):
    pl.figure()
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap=cm.coolwarm, marker='o', label='Target samples', alpha=0.2)
    pl.scatter(transp_Xs[:, 0], transp_Xs[:, 1], c=ys, cmap=cm.coolwarm, marker='+', label='Source samples', s=30)
    pl.xticks([])
    pl.yticks([])
    pl.title(f'Transported samples ({method}-Sinkhorn Transport)')
    pl.legend(loc="lower left")
    if savefig == 1:
        pl.savefig(path + filename)
    # pl.show()