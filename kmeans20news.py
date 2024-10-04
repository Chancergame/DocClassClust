import sklearn.feature_extraction.text
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans as sk_kmeans
from nltk.cluster.kmeans import KMeansClusterer as nl_kmeans
from nltk.cluster.util import cosine_distance
import numpy as np
import pandas as pd
from sklearn import metrics
from math import sqrt

file = open('20docs3UnDif.txt', 'r')
corpus = [line.strip() for line in file]
file.close()

#'comp.graphics', 'rec.autos', 'sci.med',  'talk.politics.mideast'
categories = ['talk.politics.guns', 'talk.politics.mideast',  'talk.politics.misc']
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=categories)
Y = newsgroups.target

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
ind = np.array([], dtype=int)
for i in range(len(X)):
    if np.dot(X[i], X[i]) == 0.0:
        ind = np.append(ind,i)
X = np.delete(X, ind, axis=0)
Y = np.delete(Y, ind, axis=0)

#cos_matrix = [[cos_metric(i,j) for j in X] for i in X]

iter = 30
a_rand = np.zeros((2,iter))
v_measure = np.zeros((2,iter))
mutual = np.zeros((2,iter))
fowlkes = np.zeros((2,iter))

for i in range(iter):
    eukl_pred = sk_kmeans(n_clusters=3, init='k-means++', n_init='auto').fit(X)
    eucl_pred = eukl_pred.labels_
    a_rand[0][i] = metrics.rand_score(Y, eucl_pred)
    v_measure[0][i] = metrics.v_measure_score(Y,eucl_pred)
    mutual[0][i] = metrics.adjusted_mutual_info_score(Y,eucl_pred)
    fowlkes[0][i] = metrics.fowlkes_mallows_score(Y, eucl_pred)

    nl_clusterer = nl_kmeans(3, distance=cosine_distance, avoid_empty_clusters=True)
    cos_pred = nl_clusterer.cluster(X, assign_clusters=True)
    cos_pred = np.array(cos_pred)
    a_rand[1][i] = metrics.rand_score(Y, cos_pred)
    v_measure[1][i] = metrics.v_measure_score(Y, cos_pred)
    mutual[1][i] = metrics.adjusted_mutual_info_score(Y, cos_pred)
    fowlkes[1][i] = metrics.fowlkes_mallows_score(Y, cos_pred)
    print(i)

evcl = pd.DataFrame({'a_rand': [np.min(a_rand[0]), np.mean(a_rand[0]), np.max(a_rand[0])],
        'v_measure': [np.min(v_measure[0]), np.mean(v_measure[0]), np.max(v_measure[0])],
        'mutual': [np.min(mutual[0]), np.mean(mutual[0]), np.max(mutual[0])],
        'fowlkes': [np.min(fowlkes[0]), np.mean(fowlkes[0]), np.max(fowlkes[0])]},
        index=['min', 'avrg', 'max'])
cos = pd.DataFrame({'a_rand': [np.min(a_rand[1]), np.mean(a_rand[1]), np.max(a_rand[1])],
        'v_measure': [np.min(v_measure[1]), np.mean(v_measure[1]), np.max(v_measure[1])],
        'mutual': [np.min(mutual[1]), np.mean(mutual[1]), np.max(mutual[1])],
        'fowlkes': [np.min(fowlkes[1]), np.mean(fowlkes[1]), np.max(fowlkes[1])]},
        index=['min', 'avrg', 'max'])
print(evcl)
print(cos)