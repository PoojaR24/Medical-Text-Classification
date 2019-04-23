
# coding: utf-8

# In[1]:


#Import necessary libraries
import re
import nltk
import math
import string
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
import scipy as sp
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict


# In[2]:


# open test file and read its lines
with open("train.dat", 'r') as fh:
    lines1 = fh.readlines()
with open("test.dat", 'r') as fh:
    lines2 = fh.readlines()


# In[3]:


#spliting the scentences into words
trn = [l.split() for l in lines1]
tst = [l.split() for l in lines2]


# In[4]:


#obtaining the class labels from documents
labels = []
words = []
for i in range (0,len(trn)):
    labels.append(trn[i][0])
    words.append(trn[i][1:])


# In[5]:


#Visualising the class labels for each parah
table = pd.DataFrame()

table['Labels'] = labels[:]
table['Tokens'] = words[:]

table.head()


# In[6]:


# putting the train and test data in a single document

for i in range(0,len(tst)):
    words.append(tst[i])


# In[7]:


## PREPROCESSING ##

# converting words to lowercase and remove digits
def filterInput(words):
    new_words = []
    for i in words:
        new = []
        for word in i:
            new.append(word.lower())
            for char in word:
                if(not char.isalpha()):
                    new.remove(word.lower())
                    break
        new_words.append(new)
    
    return new_words

# Remove stop words
def stop_words_remover(words):
    eng_words = set(stopwords.words('english'))
    new_words = []
    for i in words:
        new = []  
        for word in i:
            if word not in eng_words:
                new.append(word)
        new_words.append(new)
            
    return new_words

# Remove puntuations
# \s refers to the whitespace characters (which includes [ \t\n\r\f\v])
# \w includes most characters that can be part of a word in any language, as well as numbers and the underscore
def punct_remover(words):
    new_docs = []
    for i in words:
        new_words = []  
        for word in i:
            new = re.sub(r'[^\w\s]', '', word)
            if new != '':
                new_words.append(new)
        new_docs.append(new_words)
            
    return new_docs

# Remove words shorter than the minlen
def filterLen(docs, minlen):
    r""" filter out terms that are too short. 
    docs is a list of lists, each inner list is a document represented as a list of words
    minlen is the minimum length of the word to keep
    """
    return [ [t for t in d if len(t) >= minlen ] for d in docs ]

# lemmatize and keep meaningful ('verb') words
def lemmatize(docs):
    lemmatizer = WordNetLemmatizer()
    new_docs = []
    for doc in docs:
        lemmas = []  
        for word in doc:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        new_docs.append(lemmas)
            
    return new_docs


# In[8]:


# Pre-process data
words_1 = stop_words_remover(words)
words_2 = punct_remover(words_1)
words_3 = filterLen(words_2,4)
words_4 = lemmatize(words_3)
words_5 = filterInput(words_4)


# In[9]:


def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )


# In[10]:


# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat


# In[11]:


# Generating csr matrix
M = build_matrix(words_5)
print(csr_info(M))
N = csr_idf(M, copy=True)
Words = csr_l2normalize(N, copy=True)
M = build_matrix(words_5)
print(csr_info(Words))


# In[12]:


# Separating Test and Train data 
Train_words = Words[0:14438]
Test_words = Words[14438:]
Train_label = labels[0:14438]
Test_label = labels[14438:]


# In[13]:


Words.shape


# In[14]:


# Calculate Cosine Similarity
def cosine_sim(i,trainword):
    prod = i.dot(trainword.T)
    csim = list(zip(prod.indices, prod.data))
    return csim


# In[15]:


# KNN Classification
def Classification(i,train_word,train_label,k):
    
    cosim = cosine_sim(i,train_word)
    if len(cosim) == 0:
        
        if np.random.rand() > 0.5: 
            return '+'
        else:
            return '-'
    cosim.sort(key=lambda i: i[1], reverse = True)
    count = Counter(train_label[c[0]] for c in cosim[:k]).most_common(2)
    if len(count) < 2 or count[0][1] >count[1][1]:
        return count[0][0]
    count = defaultdict(float)
    for c in cosim[:k]:
        count[train_label[c[0]]] += c[1]
    return sorted(count.items(), key=lambda i: i[1], reverse = True)[0][0]


# In[17]:


Predict = []
file = open("Prediction3.dat","w+")
for i in Test_words:
    predicted_class = Classification(i,Train_words, Train_label,39)
    file.write(str(predicted_class) + "\n")
file.close()

