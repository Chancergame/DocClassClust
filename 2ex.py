import sklearn.feature_extraction.text
from numpy import dot
from numpy. linalg import norm

def cosinus_dist(x,y):
    return dot (x, y)/(norm (x) * norm (y))
def eucl_dist(x,y):
    return norm (x-y)
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
corpus = l = [line.strip() for line in file]

vectorizer = sklearn.feature_extraction.text.CountVectorizer()
X = vectorizer.fit_transform(corpus)
X = X.toarray()

#print(X[0], X[1])
print(cosinus_dist(X[0],X[1]))
print(eucl_dist(X[0],X[1]))
print(jac_dist(X[0],X[1]))