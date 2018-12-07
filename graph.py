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

nLines = 100 #Number of lines by block 
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

charsL = ["Dracula", "Jonathan", "Arthur", "Quincey", "Mina", "Renfield", "Seward", "Abraham",
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
results = np.zeros([len(charsL),len(charsL)]) #init tab
#single block ?
for block in blocksStemmed:
    for charOne in chars:
        for charTwo in chars:
            if (charOne in block and charTwo in block) and charOne != charTwo: #if the two chars are in the block then add 1 to their relation 
                results[chars[charOne], chars[charTwo]] = results[chars[charOne], chars[charTwo]] + 1

#The diagonal of the matrix tells us how many times a character had been mentionned in the book. 
###########################
#Without even reading the book, a lot of information can be extracted from this matrix. 
#First off, the main character isn't dracula as the name of the book suggest, but the book is more about Harker and van helsing, as 
#show the column 1 and column 12.
#Second, Dracula is very absent from the books, he has some connections with Jonathan, Mina, Seward and Van helsing (who is tracking him)
#But it doesn't seem like he is omnipresent. 
#Third, it seems like mina have some relations with Dracula but her name occures more often with Jonathan, Van helsing, Lucy and Seward.
#This is partly because those characters talk a lot about Mina but a more interesting thing can be pointed here. 
#The fact that Dracula doesn't mention her very often is in fact mostly due to the writing style of the author and the personality of the character...
#How ? This is quite simple, Dracula often refer as Mine as her "loved one", or "my dear" or these kind of formulations which both informs us on the character style (after verification) 
#But also on a limitation of our method in this particular context. It is hard to make the algorithm understand that Dracula is in fact talking about Mina when he uses so many different nicknames. 
###########################               
import networkx as nx

G = nx.from_numpy_matrix(results)















