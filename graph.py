#############################################
#Now building a graph representing the book 
#We will read the file and separate it every 10 lines 
#Then we will check which ones of the principal characters appear 
#We can then use this relation between characters appearing together to build a frequency of "appearing together" 
#Then this will be used to build a graph seeing how do the characters interact in the book 
#Eventually we could try using pagerank in order to see which characters are the most important ones (obciously dracula should be first here..)
#############################################



filepath = "C:/Users/Lucien/Desktop/Dracula/dracula.txt"

#Open file and divide into blocks
###########

nLines = 50 #Number of lines by block 
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

#################################
#Now to make our data manipulation easier we are going to associate main characters with a number in a dictionary, so we can make a matrix and use names as indices
#for clarity in code. 

charsL = ["Dracula", "Jonathan", "Arthur", "Quincey", "Mina", "Wilhelmina", "Renfield", "John", "Jack", "Seward", "Abraham",
          "Van Helsing", "Lucy"]
#John = Jack = Steward
#Mina = Wilhelmina
#Abraham = van helsing

#Now stemming on these as well so they are equal to the stemmed text blocks. 

counter = 0
for word in charsL:
    charsL[counter] = charsL[counter].lower()
    charsL[counter] = snow.stem(charsL[counter])
    counter +=1

chars = {}
counter = 0
for word in charsL:
    chars[word] = counter
    counter +=1
    
n = len(chars)
#Now create the adjacency matrix (take frequency into account)
#Matrix to stock results .. Then we can add results of synonyms 
results = np.zeros([13,13]) #init tab
#single block ?
for block in blocksStemmed:
    for charOne in chars:
        for charTwo in chars:
            if (charOne in block and charTwo in block) and charOne != charTwo: #if the two chars are in the block then add 1 to their relation 
                results[chars[charOne], chars[charTwo]] = results[chars[charOne], chars[charTwo]] + 1

#The diagonal of the matrix tells us how many times a character had been mentionned in the book. 
                
import networkx as nx

G = nx.from_numpy_matrix(results)















