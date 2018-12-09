#############################################
#Now building a graph representing the book 
#We will read the file and separate it every 10 lines 
#Then we will check which ones of the principal characters appear 
#We can then use this relation between characters appearing together to build a frequency of "appearing together" 
#Then this will be used to build a graph seeing how do the characters interact in the book 
#Eventually we could try using pagerank in order to see which characters are the most important ones (obciously dracula should be first here..)
#############################################

#DISCLAIMER : To run this part you can run the full script. Outputs a graph. charPR are the values for pagerank, results is the matrix of "adjacency"  


filepath = "C:/Users/Lucien/Desktop/Dracula/dracula.txt"

#Open file and divide into blocks
###########
import nltk as nltk
sentences = []

with open(filepath, 'r') as myfile:
    data=myfile.read().replace('\n', ' ')

sentences = nltk.tokenize.sent_tokenize(data)

nSent = 15
blocks = []
temp = ""
counter = 0
for i in range(0, len(sentences)):
    temp = temp + " " + sentences[i]
    counter+=1
    if counter == nSent:
        counter = 0
        blocks.append(temp)
        temp = ""

blocksOriginals = blocks.copy()

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

charsL = ["Dracula", "Jonathan", "Arthur", "Quincey", "Mina", "Renfield", "Seward",
          "Van Helsing", "Lucy"]

charsLabels = ["Dracula", "Jonathan", "Arthur", "Quincey", "Mina", "Renfield", "Seward",
          "Van Helsing", "Lucy"]


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
#Pagerank algorithm : some parts reused from my own work on the labs 
# 1.2 #######################################
#First is iterative method, then exact method

resultsTwo = np.array(results)
for i in range(0, len(charsL)):
    for j in range(0, len(charsL)):
        if resultsTwo[i][j] < 10:
            resultsTwo[i][j] = 0
G = nx.from_numpy_matrix(resultsTwo)

import numpy as np 
import networkx as nx
from scipy import linalg

#G is the generated graph
#S is the matrix representing links

#Reset G to the matrix without treshold
G = nx.from_numpy_matrix(results)


def pageRankIt(G, alpha = 0.85, K = 2000):
    nNodes = nx.number_of_nodes(G)

    #Creating the network
    S = nx.to_numpy_matrix(G) #Numpy adj matrix
    
    for i in range(0, nNodes): 
        S[i, :] = S[i, :] / np.sum(S[i, :])
    summ = np.sum(S, axis = 1) #check == 1, probability that surfer goes from i to j
    
    v = np.random.rand(nNodes, 1) #initial guess
    v = v / np.linalg.norm(v, 1) #L1
    
    #google matrix 
    GM = (alpha * S) + (((1 - alpha) / nNodes) * np.matmul(np.ones(nNodes), v))
    summG = np.sum(GM, axis = 1) #check == 1, stochastic matrix
    
    for i in range(0, nNodes):
        GM[i, :] = GM[i, :] / np.sum(GM[i, :])
    
    v = v.transpose()
    
    for i in range(0, K):
        v = np.matmul(v, GM)
        
    v = v.transpose() #put it back in column
    
    #Normalize vector by its norm
    v = v/np.linalg.norm(v, 1)

    return v

v = pageRankIt(G)

# =============================================================================
# Pagerank is originally an algorithm used by google to generate a ranking between web pages. It mesures the poopularity of a web page but pagerank is not the only 
# algorithm used in this case, it is just one indicator used to order the results of a research.
# 
# It works by assigning each page a score proportional to how many times an user would get on the said page by cliquing on random links all pages. 
# Therefore a page is linked by a lot of other pages is going to have an high score. We can say that pagerank is a random walk on the graph. 
# The output is a probability distribution of the likelihoof that the user gets on a said page by randomly clicking. 
# =============================================================================
# =============================================================================
# The algorithm therefore calculates which pages are the most "central" ones, in our case it can be interpreted as 
# which characters are the centrals ones, and are connected most to other main characters. 
# It means that we can use it to make a ranking of what are the most important characters troughout the story. 
# =============================================================================

#Now we are going to assign scores to characters into a dictionary. 

charPR = {}
counter = 0
for i in range(0, len(chars)):
    charPR[charsLabels[counter]] = v[counter][0,0]
    counter +=1

#ordering
import operator
orderedPR = sorted(charPR.items(), key=operator.itemgetter(1))

print("Pagerank done, now printing character from most connected ones to less connected ones ...")
for i in range(8, -1, -1):
    print(orderedPR[i][0], " : ", orderedPR[i][1])
    
# =============================================================================
# Advantages : we can represent most importants characters of the story very easily, it is a fast and iterative algorithm that converges to a solution. 
# Cons : As we can see, if a character is not directly linked to other characters it won't have a high score. This is easily explained 
# by the fact that pagerank is based on the links between characters. But in our story Dracula is the main "bad guy" pf the story and yet is ranked last... 
# Why ? Well it is simply because he rarely appears in the book and therefore is not connected to a lot of other characters. 
# So if a character is important but stays away from the other main characters it will not be ranked as high as he normally should be. 
# Altough this gives us a good estimation on the written text itself, informing us that dracula is not appearing a lot in the book.  
# =============================================================================

#Now the same graph with scores as labels ...
#Mapping labels
labels = {}
counter = 0
for label in orderedPR:
    labels[counter] = label[0] +" : " + str(round(label[1], 4))
    counter +=1
    

#Using graph with treshold for representation, much clearer 
G = nx.from_numpy_matrix(resultsTwo)
    
H=nx.relabel_nodes(G,labels)
nx.draw(H, with_labels = True)














