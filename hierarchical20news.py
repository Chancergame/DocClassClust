import sklearn.feature_extraction.text
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

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

a_rand = np.zeros((2,4))
v_measure = np.zeros((2,4))
mutual = np.zeros((2,4))
fowlkes = np.zeros((2,4))

linkage = ['ward', 'complete', 'average', 'single']
for i,link in enumerate(linkage):
    hierachical = AgglomerativeClustering(n_clusters=3, linkage=link).fit(X)
    eucl_pred = hierachical.labels_
    a_rand[0][i] = metrics.rand_score(Y, eucl_pred)
    v_measure[0][i] = metrics.adjusted_rand_score(Y, eucl_pred)
    mutual[0][i] = metrics.homogeneity_score(Y, eucl_pred)
    fowlkes[0][i] = metrics.completeness_score(Y, eucl_pred)

linkage = ['complete', 'average', 'single']
for i,link in enumerate(linkage):
    hierachical = AgglomerativeClustering(n_clusters=3, linkage=link, metric='cosine').fit(X)
    eucl_pred = hierachical.labels_
    a_rand[1][i] = metrics.rand_score(Y, eucl_pred)
    v_measure[1][i] = metrics.adjusted_rand_score(Y, eucl_pred)
    mutual[1][i] = metrics.homogeneity_score(Y, eucl_pred)
    fowlkes[1][i] = metrics.completeness_score(Y, eucl_pred)

eucl = pd.DataFrame({'a_rand': a_rand[0],
        'v_measure': v_measure[0],
        'mutual': mutual[0],
        'fowlkes': fowlkes[0]},
        index=['ward', 'complete', 'average', 'single'])
cos = pd.DataFrame({'a_rand': a_rand[1],
        'v_measure': v_measure[1],
        'mutual': mutual[1],
        'fowlkes': fowlkes[1]},
        index=['complete', 'average', 'single', 'ward'])
print(eucl)
print(cos)