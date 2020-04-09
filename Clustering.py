# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:37:15 2019

@author: miche
"""

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


def clusteringKMeans(X, i):
    kmeansAtt = KMeans(n_clusters=i, max_iter=300, init='k-means++', random_state=12345).fit(X)

    return kmeansAtt.cluster_centers_


def clusteringMiniBatchKMeans(X, i, dim):
    kmeansAtt = MiniBatchKMeans(n_clusters=i, max_iter=300, init='k-means++', random_state=12345, init_size=dim,
                                batch_size=50).fit(X)

    return kmeansAtt.cluster_centers_
