import sklearn.feature_extraction.text
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from nltk.cluster.kmeans import KMeansClusterer
import nltk

from pyclustering.cluster.kmeans import kmeans as kmeans_pc
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from sklearn import metrics

def jac_dist(x,y):
    x1 = set()
    y1 = set()
    for i in range(len(x)):
        if x[i] > 0:
           x1.add(i)
    for i in range(len(y)):
        if y[i] > 0:
           y1.add(i) 
    return len(x1.intersection(y1))/len(x1.union(y1))

file = open('2.txt', 'r')
corpus = [line.strip() for line in file]

vectorizer = sklearn.feature_extraction.text.CountVectorizer()
X2 = vectorizer.fit_transform(corpus)
X2 = X2.toarray()

true = np.array([ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
#0,

iter = 50
rand = np.zeros((3,iter))
adjusted_rand = np.zeros((3,iter))
homogeneity = np.zeros((3,iter))
completeness = np.zeros((3,iter))
for i in range(iter):
    kmeans_sk = KMeans(n_clusters=2, init='k-means++').fit(X2)
    eucl_pred = kmeans_sk.labels_
    rand[0][i] = metrics.rand_score(true, eucl_pred)
    adjusted_rand[0][i] = metrics.adjusted_rand_score(true, eucl_pred)
    homogeneity[0][i] = metrics.homogeneity_score(true, eucl_pred)
    completeness[0][i] = metrics.completeness_score(true, eucl_pred)

    kclusterer = KMeansClusterer(2, distance=nltk.cluster.util.cosine_distance)
    assigned_clusters = kclusterer.cluster(X2, assign_clusters=True)
    cos_pred = np.array(assigned_clusters)
    rand[1][i] = metrics.rand_score(true, cos_pred)
    adjusted_rand[1][i] = metrics.adjusted_rand_score(true, cos_pred)
    homogeneity[1][i] = metrics.homogeneity_score(true, cos_pred)
    completeness[1][i] = metrics.completeness_score(true, cos_pred)

    jac_metric = distance_metric(type_metric.USER_DEFINED, func=jac_dist)
    initial_centers = kmeans_plusplus_initializer(X2, 2).initialize()
    kmeans_instance = kmeans_pc(X2, initial_centers, metric=jac_metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    jac_pred = np.zeros(len(true), dtype=int)
    for i in range(len(clusters)):
        for j in clusters[i]:
            jac_pred[j] = i
    rand[2][i] = metrics.rand_score(true, jac_pred)
    adjusted_rand[2][i] = metrics.adjusted_rand_score(true, jac_pred)
    homogeneity[2][i] = metrics.homogeneity_score(true, jac_pred)
    completeness[2][i] = metrics.completeness_score(true, jac_pred)

evcl = pd.DataFrame({'rand': [np.min(rand[0]), np.mean(rand[0]), np.max(rand[0])],
        'a_rand': [np.min(adjusted_rand[0]), np.mean(adjusted_rand[0]), np.max(adjusted_rand[0])],
        'hom': [np.min(homogeneity[0]), np.mean(homogeneity[0]), np.max(homogeneity[0])],
        'comp': [np.min(completeness[0]), np.mean(completeness[0]), np.max(completeness[0])]},
        index=['min', 'avrg', 'max'])
cos = pd.DataFrame({'rand': [np.min(rand[1]), np.mean(rand[1]), np.max(rand[1])],
        'a_rand': [np.min(adjusted_rand[1]), np.mean(adjusted_rand[1]), np.max(adjusted_rand[1])],
        'hom': [np.min(homogeneity[1]), np.mean(homogeneity[1]), np.max(homogeneity[1])],
        'comp': [np.min(completeness[1]), np.mean(completeness[1]), np.max(completeness[1])]},
        index=['min', 'avrg', 'max'])
jac = pd.DataFrame({'rand': [np.min(rand[2]), np.mean(rand[2]), np.max(rand[2])],
        'a_rand': [np.min(adjusted_rand[2]), np.mean(adjusted_rand[2]), np.max(adjusted_rand[2])],
        'hom': [np.min(homogeneity[2]), np.mean(homogeneity[2]), np.max(homogeneity[2])],
        'comp': [np.min(completeness[2]), np.mean(completeness[2]), np.max(completeness[2])]},
        index=['min', 'avrg', 'max'])
print(evcl)
print(cos)
print(jac)