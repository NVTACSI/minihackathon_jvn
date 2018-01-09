#python example to infer document vectors from trained doc2vec model
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import pandas as pd
import gensim.models as g
import codecs
import os
import csv
import numpy as np 
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def preprocess(doc):
    punc_free = ''.join(ch for ch in doc if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
#    doc_filter = " ".join([word for word, pos in nltk.pos_tag(normalized.lower().split()) if pos=='NN' or 'VB' in pos])
    stop_free = ' '.join([i for i in normalized.split() if i not in stop])

#    doc_count = pd.Series(stop_free).value_counts()
#    doc_count = set(doc_count[doc_count==1].index.values)
#    new_words = ' '.join([w for w in stop_free if w not in doc_count])
#    stop_free = doc.strip().split()    
    return stop_free

#parameters
model="doc2vec.bin"
path = 'inaugural_speeches/'
files = [path+fname for fname in os.listdir(path)]
files = sorted(files)
m = g.Doc2Vec.load(model)

output_file="test_vectors.txt"
docvectors = []
for fname in files:
    print "processing {}".format(fname)
    #inference hyper-parameters
    start_alpha=0.01
    infer_epoch=1000

    #load model
    f = open(fname, 'r')
    doc = ' '.join(f.readlines())
    doc = preprocess(doc)

    #infer test vectors
    vector = m.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    docvectors.append(list(vector))

np.savetxt('docvector.csv', docvectors)
