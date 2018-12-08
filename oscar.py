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

# Open file and divide into chapters ###########

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

# BAG OF WORDS ######################### 

bowChapters = []
for i in range(0,27):
    countV = CountVectorizer(max_features=50) #top 100 words for the chapter 
    chapter = [temp2[i]]
    bowChapters.append(countV.fit_transform(chapter))
    
print(bowChapters[2])
temp[2][1]

# Word cloud #########################

import numpy as np
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import os
import random

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % 0


# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = "D:/Google Drive/DATS2MS/LINMA2472 Algorithms in data science/Project/Dracula"

# read the mask image taken from
# http://www.stencilry.org/stencils/movies/star%20wars/storm-trooper.gif
mask = np.array(Image.open(path.join(d, "drac.png")))

# Texte
text = open(path.join(d, 'dracula.txt')).read()

# Stopwords
stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("must")
stopwords.add("will")
stopwords.add("one")
stopwords.add("n't")
stopwords.add("may")

tokens = np.array(nltk.word_tokenize(text))
tagged = np.array(nltk.pos_tag(tokens))

isnt_verb = np.empty(shape = (tokens.size), dtype = bool)
for i in range(0, tokens.size-1):
    if (tagged[i,1] != "VB" and tagged[i,1] != "VBD" and tagged[i,1] != "VBG"
        and tagged[i,1] != "VBN" and tagged[i,1] != "VBP" and tagged[i,1] != "VBZ"):
        isnt_verb[i] = True
    else:
        isnt_verb[i] = False
        
text_noverb = tokens[(isnt_verb)]
text_niverbstr = " ".join(text_noverb)

# Create wordcloud
wc = WordCloud(background_color="white", max_words=1000, mask=mask, stopwords=stopwords, margin=10,
               random_state=1).generate(text_niverbstr)
default_colors = wc.to_array()
plt.title("Custom colors")
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
wc.to_file("D:/Google Drive/DATS2MS/LINMA2472 Algorithms in data science/Project/Dracula/wordcloud.png")
plt.axis("off")
plt.show()






