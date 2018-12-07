# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:56:15 2018

@author: Lucien
"""

# =============================================================================
# #Reading file
# 
# 
# startStops = [("CHAPTER I", "CHAPTER II"), ("CHAPTER II", "CHAPTER III"), ("CHAPTER III", "CHAPTER IV"), ("CHAPTER IV", "CHAPTER V"), ("CHAPTER V", "CHAPTER VI"), 
#           ("CHAPTER VI", "CHAPTER VII"), ("CHAPTER VII", "CHAPTER VIII"), ("CHAPTER VIII", "CHAPTER IX"), ("CHAPTER IX", "CHAPTER X"), ("CHAPTER X", "CHAPTER XI"), 
#           ("CHAPTER XI", "CHAPTER XII"), ("CHAPTER XII", "CHAPTER XIII"), ("CHAPTER XIII", "CHAPTER XIV"), ("CHAPTER XIV", "CHAPTER XV"), ("CHAPTER XV", "CHAPTER XVI"), 
#           ("CHAPTER XVI", "CHAPTER XVII"), ("CHAPTER XVII", "CHAPTER XVIII"), ("CHAPTER XVIII", "CHAPTER XIX"), ("CHAPTER XIX", "CHAPTER XX"), ("CHAPTER XX", "CHAPTER XXI"), 
#           ("CHAPTER XXI", "CHAPTER XXII"), ("CHAPTER XXII", "CHAPTER XXIII"), ("CHAPTER XXIII", "CHAPTER XXIV"), ("CHAPTER XXIV", "CHAPTER XXV"), ("CHAPTER XXV", "CHAPTER XXVI"),
#           ("CHAPTER XXVI", "CHAPTER XXVII"), ("CHAPTER XXVII", "THE END")]
# 
# chapters = []
# 
# filepath = "C:/Users/Lucien/Desktop/Dracula/dracula.txt"
# 
# #Open file and divide into chapters
# ###########
# with open(filepath, 'r') as myfile:
#     data=myfile.read().replace('\n', ' ')
# 
# import re
# for i in range(0, 27):
#     Start = startStops[i][0]
#     End = startStops[i][1]
#     expression = re.compile(r'%s.*?%s' % (Start,End), re.S)
#     chapters.append(expression.search(data).group(0)) 
# 
# filepath = "C:/Users/Lucien/Desktop/Dracula/dracula.txt"
# 
# #Open file and divide into blocks
# ###########
# 
# =============================================================================

filepath = "C:/Users/Lucien/Desktop/Dracula/dracula.txt"

nLines = 15 #Number of lines by block 
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
        
blocksStemmed = temp2

########################################
#This section aims to use the tfidf similarity cosine in order to make a simple search engine
#The user fixes the query and the goal is to find the most similar chapter. 
#For exemple the first chapter talks about diner and visiting, so our exemple query will return that the first chapter is the most 
#similar to our query and therefore is the one we are looking for.
#If you used "Dracula is dead and sleeping" then it would return the last chapter. 
#
#To use this just change the query by a sentence of your choice (better if related to the book). 
########################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

query = "I stopped at hotel royale, we had a good diner and i then visited a museum."
cleanr = re.compile('<.*?>')
query = query.lower()
query = re.sub(cleanr, ' ', query) #HTML tags
query = re.sub(r'[?|!|\'|"|#]',r'',query)
query = re.sub(r'[.|,|)|(|\|/|_|-|;]',r' ',query) #Punctuations

#Stem query
snow = nltk.stem.SnowballStemmer("english") #stemmer
query = [snow.stem(word) for word in query.split() if word not in stop]  #Stop is stopwords set

#Back to string 
queryStemmed = ''
chapterUniqueTemp = temp[i]
for word in query:
    queryStemmed = queryStemmed + ' ' + word
query = queryStemmed        

#Now the query is preprocessed ...

#Query added as first of the list, first of the list is the query for following code
queryChaps = []
queryChaps.append(query)
queryChaps.extend(blocksStemmed)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(queryChaps)  #tfidf score with normalization (we take frequencies into account)

scores = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train) #here the first element of tfidf matrix, since query is the first element of list
#convert to list
scores = scores.tolist()
scores = scores[0]
#change score of query since it is =1 and would prevent next line from working properly
scores[0] = 0
maxScore = max(scores)
maxPos = [i for i, j in enumerate(scores) if j == maxScore] #!= 1 to avoid duplicates

print("Tf-idf cosine scores : ", scores, "\n\n")
print("The block you are looking for should be block", maxPos, "\n\n")
print("---------------------------------------")
print("Block : ", blocksOriginals[maxPos[0]-1])







