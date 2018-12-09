# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 19:13:34 2018

@author: Lucien
"""

#Topic extraction, NMF and LDA
#Reading file


startStops = [("CHAPTER I", "CHAPTER II"), ("CHAPTER II", "CHAPTER III"), ("CHAPTER III", "CHAPTER IV"), ("CHAPTER IV", "CHAPTER V"), ("CHAPTER V", "CHAPTER VI"), 
          ("CHAPTER VI", "CHAPTER VII"), ("CHAPTER VII", "CHAPTER VIII"), ("CHAPTER VIII", "CHAPTER IX"), ("CHAPTER IX", "CHAPTER X"), ("CHAPTER X", "CHAPTER XI"), 
          ("CHAPTER XI", "CHAPTER XII"), ("CHAPTER XII", "CHAPTER XIII"), ("CHAPTER XIII", "CHAPTER XIV"), ("CHAPTER XIV", "CHAPTER XV"), ("CHAPTER XV", "CHAPTER XVI"), 
          ("CHAPTER XVI", "CHAPTER XVII"), ("CHAPTER XVII", "CHAPTER XVIII"), ("CHAPTER XVIII", "CHAPTER XIX"), ("CHAPTER XIX", "CHAPTER XX"), ("CHAPTER XX", "CHAPTER XXI"), 
          ("CHAPTER XXI", "CHAPTER XXII"), ("CHAPTER XXII", "CHAPTER XXIII"), ("CHAPTER XXIII", "CHAPTER XXIV"), ("CHAPTER XXIV", "CHAPTER XXV"), ("CHAPTER XXV", "CHAPTER XXVI"),
          ("CHAPTER XXVI", "CHAPTER XXVII"), ("CHAPTER XXVII", "THE END")]

chapters = []

filepath = "C:/Users/Lucien/Desktop/Dracula/dracula.txt"

#Open file and divide into chapters
###########
with open(filepath, 'r') as myfile:
    data=myfile.read().replace('\n', ' ')

import re
for i in range(0, 27):
    Start = startStops[i][0]
    End = startStops[i][1]
    expression = re.compile(r'%s.*?%s' % (Start,End), re.S)
    chapters.append(expression.search(data).group(0)) 

###########
#Preprocessing
###########

import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer
import scipy as sp
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from gensim.models import Word2Vec    
  
nltk.download('stopwords') #Stopwords list             

#stopwords
stop = set(stopwords.words('english'))

#basic text preprocessing
for i in range(0, 27):
    chapterUnique = chapters[i]
    cleanr = re.compile('<.*?>')
    chapterUnique = chapterUnique.lower()
    chapterUnique = re.sub(cleanr, ' ', chapterUnique) #HTML tags
    chapterUnique = re.sub(r'[?|!|\'|"|#]',r'',chapterUnique)
    chapterUnique = re.sub(r'[.|,|)|(|\|/|_|-|;]',r' ',chapterUnique) #Punctuations
    chapters[i] = chapterUnique

temp = []
wordsByChapter = []
snow = nltk.stem.SnowballStemmer("english") #stemmer
for i in range(0, 27):
    chapterUnique = chapters[i]
    words = [snow.stem(word) for word in chapterUnique.split() if word not in stop]
    temp.append(words)
    
    
#Convert back to string
temp2 = []
for i in range(0,27):
    chapterStemmed = ''
    chapterUniqueTemp = temp[i]
    for word in chapterUniqueTemp:
        chapterStemmed = chapterStemmed + ' ' + word
    temp2.append(chapterStemmed)
        
chaptersStemmed = temp2

#BAG OF WORDS######################### 

bowChapters = []
for i in range(0,27):
    countV = CountVectorizer(max_features=50) #top 100 words for the chapter 
    chapter = [temp2[i]]
    bowChapters.append(countV.fit_transform(chapter))
    

###############
from sklearn.decomposition import NMF, LatentDirichletAllocation
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#LDA : It uses the tf vectorizer 
nFeatures = 1000
nTopics = 1

for j in range(0,27):
    warnings.filterwarnings("ignore")
    tf_vectorizer = CountVectorizer( max_features=nFeatures, stop_words='english')
    tf = tf_vectorizer.fit_transform([chaptersStemmed[j]])
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    lda = LatentDirichletAllocation(n_topics=nTopics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    
    def topics(lda, features, nWords):
        for topic_idx, topic in enumerate(lda.components_):
            print("\n")
            print("Topic ", str(j+1), ": ", end = "")
            for i in topic.argsort()[:-nWords -1 : -1]:
                print("".join([features[i]]), end = " ")
    
    topics(lda, tf_feature_names, 10)
    
#NMF
from sklearn.decomposition import NMF, LatentDirichletAllocation
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

nFeatures = 1000
nTopics = 1

for j in range(0,27):
    warnings.filterwarnings("ignore")
    tfidf_vectorizer = TfidfVectorizer(max_features=nFeatures, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform([chaptersStemmed[j]])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    
    nmf = NMF(n_components=nTopics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)    
    def topics(lda, features, nWords):
        for topic_idx, topic in enumerate(lda.components_):
            print("\n")
            print("Topic ", str(j+1), ": ", end = "")
            for i in topic.argsort()[:-nWords -1 : -1]:
                print("".join([features[i]]), end = " ")
    
    topics(nmf, tf_feature_names, 10)

#Original topic extraction : full book divided by chapters
    

filepath = "C:/Users/Lucien/Desktop/Dracula/dracula.txt"

nLines = 100000 #Number of lines by block 
blocks = [] #blocks of lines 
counter = 0 #counter for lines 
cacheBlock = "" #Cache for block 
for line in open(filepath, 'r'):
    if line != "\n":
        #delete the \n, also add space between lines
        line = [line[:-1]]
        line = "".join(line)
        cacheBlock = cacheBlock + " " + line
        counter +=1

    if counter == nLines:
        blocks.append(cacheBlock)
        cacheBlock = ""
        counter = 0
blocks.append(cacheBlock)

blocksOriginals = list(blocks)
#Now we are going to stem and remove punctuations, stopwords. 

###########
#Preprocessing (code reused from tfidf code, so var names are a bit misleaging but the output we care about is "blocksStemmed")
###########
        
import re
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer
import scipy as sp
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from gensim.models import Word2Vec    
  
nltk.download('stopwords') #Stopwords list             

#stopwords
stop = set(stopwords.words('english'))
blockL = len(blocks)
#basic text preprocessing
for i in range(0, blockL):
    chapterUnique = blocks[i]
    cleanr = re.compile('<.*?>')
    chapterUnique = chapterUnique.lower()
    chapterUnique = re.sub(cleanr, ' ', chapterUnique) #HTML tags
    chapterUnique = re.sub(r'[?|!|\'|"|#]',r'',chapterUnique)
    chapterUnique = re.sub(r'[.|,|)|(|\|/|_|-|;]',r' ',chapterUnique) #Punctuations
    blocks[i] = chapterUnique

temp = []
wordsByChapter = []
snow = nltk.stem.SnowballStemmer("english") #stemmer
for i in range(0, blockL):
    chapterUnique = blocks[i]
    words = [snow.stem(word) for word in chapterUnique.split() if word not in stop]
    temp.append(words)
    
    
#Convert back to string
temp2 = []
for i in range(0,blockL):
    chapterStemmed = ''
    chapterUniqueTemp = temp[i]
    for word in chapterUniqueTemp:
        chapterStemmed = chapterStemmed + ' ' + word
    temp2.append(chapterStemmed)
        
bookStemmed = temp2
    
#LDA : It uses the tf vectorizer 
nFeatures = 1000
nTopics = 3

warnings.filterwarnings("ignore")
tf_vectorizer = CountVectorizer( max_features=nFeatures, stop_words='english')
tf = tf_vectorizer.fit_transform(chaptersStemmed)
tf_feature_names = tf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_topics=nTopics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

def topics(lda, features, nWords):
    for topic_idx, topic in enumerate(lda.components_):
        print("\n")
        print("Topic", topic_idx, " : ", end = "")
        for i in topic.argsort()[:-nWords -1 : -1]:
            print("".join([features[i]]), end = " ")

topics(lda, tf_feature_names, 20)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


