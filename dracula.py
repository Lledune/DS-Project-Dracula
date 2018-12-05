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
nltk.download('stopwords')                             

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
    
print(bowChapters[0])


bowChapters[0].tobsr() 
#♣OSCAR : Pour le bagofwords, si tu vois : (0,93) 23 par exemple, cela veut dire que me 93ème mot apparait 23 fois.
#Tu peux voir de quel mot il s'agit en utilisant la liste temp comme ceci 
temp[0][93] #word = "said"
sp.shape(bowChapters[0])
countV
        
        
        
        
        
