import sklearn.feature_extraction.text
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import pandas as pd

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
rand = np.zeros(3)
adjusted_rand = np.zeros(3)
homogeneity = np.zeros(3)
completeness = np.zeros(3)

hierachical = AgglomerativeClustering(n_clusters=2).fit(X2)
eucl_pred = hierachical.labels_
rand[0] = metrics.rand_score(true, eucl_pred)
adjusted_rand[0] = metrics.adjusted_rand_score(true, eucl_pred)
homogeneity[0] = metrics.homogeneity_score(true, eucl_pred)
completeness[0] = metrics.completeness_score(true, eucl_pred)


hierachical = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='complete')
hierachical.fit(X2)
cos_pred = hierachical.labels_
rand[1] = metrics.rand_score(true, cos_pred)
adjusted_rand[1] = metrics.adjusted_rand_score(true, cos_pred)
homogeneity[1] = metrics.homogeneity_score(true, cos_pred)
completeness[1] = metrics.completeness_score(true, cos_pred)

jac_matrix = np.zeros((len(X2), len(X2)))
for i in range(len(X2)):
    for j in range(len(X2)):
        jac_matrix[i][j] = jac_dist(X2[i], X2[j])
hierachical = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='complete')
hierachical.fit(jac_matrix)
jac_pred = hierachical.labels_
rand[2] = metrics.rand_score(true, jac_pred)
adjusted_rand[2] = metrics.adjusted_rand_score(true, jac_pred)
homogeneity[2] = metrics.homogeneity_score(true, jac_pred)
completeness[2] = metrics.completeness_score(true, jac_pred)

data = pd.DataFrame({'rand': rand,
                     'a_rand': adjusted_rand,
                     'hom': homogeneity,
                     'com': completeness},
                     index= ['evcl', 'cos', 'jac'])
print(data)