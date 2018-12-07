#Reading file


startStops = [("CHAPTER I", "CHAPTER II"), ("CHAPTER II", "CHAPTER III"), ("CHAPTER III", "CHAPTER IV"),
              ("CHAPTER IV", "CHAPTER V"),
              ("CHAPTER V", "CHAPTER VI"), ("CHAPTER VI", "CHAPTER VII"), ("CHAPTER VII", "CHAPTER VIII"),
              ("CHAPTER VIII", "CHAPTER IX"), ("CHAPTER IX", "CHAPTER X"), ("CHAPTER X", "CHAPTER XI"),
              ("CHAPTER XI", "CHAPTER XII"), ("CHAPTER XII", "CHAPTER XIII"), ("CHAPTER XIII", "CHAPTER XIV"),
              ("CHAPTER XIV", "CHAPTER XV"), ("CHAPTER XV", "CHAPTER XVI"), ("CHAPTER XVI", "CHAPTER XVII"),
              ("CHAPTER XVII", "CHAPTER XVIII"), ("CHAPTER XVIII", "CHAPTER XIX"), ("CHAPTER XIX", "CHAPTER XX"),
              ("CHAPTER XX", "CHAPTER XXI"), ("CHAPTER XXI", "CHAPTER XXII"), ("CHAPTER XXII", "CHAPTER XXIII"),
              ("CHAPTER XXIII", "CHAPTER XXIV"), ("CHAPTER XXIV", "CHAPTER XXV"), ("CHAPTER XXV", "CHAPTER XXVI"),
              ("CHAPTER XXVI", "CHAPTER XXVII"), ("CHAPTER XXVII", "THE END")]

chapters = []

filepath = "D:/Google Drive/DATS2MS/LINMA2472 Algorithms in data science/Project/Dracula/dracula.txt"

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
    
print(bowChapters[0])


#♣OSCAR : Pour le bagofwords, si tu vois : (0,93) 23 par exemple, cela veut dire que me 93ème mot apparait 23 fois.
#Tu peux voir de quel mot il s'agit en utilisant la liste temp comme ceci 
temp[0][93] #word = "said"

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

#Now the query is preprocesed ...

#Query added as first of the list, first of the list is the query for following code
queryChaps = []
queryChaps.append(query)
queryChaps.extend(chaptersStemmed)

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

print("Tf-idf cosine scores : ", scores)
print("The chapter you are looking for should be Chapter", maxPos)










