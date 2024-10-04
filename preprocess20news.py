from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return [tag_dict.get(tag, wordnet.NOUN), tag]
                        
#'comp.graphics', 'rec.autos', 'sci.med',  'talk.politics.mideast'
categories = ['talk.politics.guns', 'talk.politics.mideast',  'talk.politics.misc']
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=categories)
newsdata = newsgroups.data 
corpus = [i.lower() for  i in newsdata]
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
corpus = [tokenizer.tokenize(i) for i in corpus]


stopw = nltk.corpus.stopwords.words('english')
f_corpus = [[j for j in i if j not in stopw] for i in corpus]

lemmatizer = nltk.stem.WordNetLemmatizer()
lem_corpus = [[lemmatizer.lemmatize(j) for j in i if nltk.pos_tag([j])[0][1][0].upper() in 'NJ'] for i in f_corpus]
#

file = open("20docs2UnDif.txt", "w")
for i in range(len(lem_corpus)-1):
    file.write(' '.join(lem_corpus[i]) + '\n')
file.write(' '.join(lem_corpus[-1]))
file.close()