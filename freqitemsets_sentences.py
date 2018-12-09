from collections import Counter
from itertools import combinations, chain
import re
import nltk
from nltk.corpus import stopwords

# Importing data

filepath = "D:/Google Drive/DATS2MS/LINMA2472 Algorithms in data science/Project/Dracula/dracula.txt"
    
sentences = []

with open(filepath, 'r') as myfile:
    data=myfile.read().replace('\n', ' ')

sentences = nltk.tokenize.sent_tokenize(data)

# Stopwords
    
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# Basic text preprocessing

for i in range(0, len(sentences)):
    sentence = sentences[i]
    cleanr = re.compile('<.*?>')
    sentence = sentence.lower()
    sentence = re.sub(cleanr, ' ', sentence)
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/|_|-|;]',r' ',sentence)
    sentences[i] = sentence

wordssent = []
for i in range(0, len(sentences)):
    sentence = sentences[i]
    words = [word for word in sentence.split() if word not in stop]
    wordssent.append(words)
    
# Defining characters

charlist = ["count", "dracula", "jonathan", "mina", "arthur", "holwood",
             "quincey", "morris", "renfield", "john", "seward", "abraham",
             "helsing", "lucy", "westenra"]

# Keeping characters

charbysent = []
for i in range(0, len(wordssent)):
    charinsent = []
    for j in range(0, len(wordssent[i])):
        if (wordssent[i][j] in charlist):
            charinsent.append(wordssent[i][j])
    if charinsent == []: continue
    charbysent.append(charinsent)

# Replacing the names of characters
    
for i in range(0, len(charbysent)):
    for j in range(0, len(charbysent[i])):
        if charbysent[i][j] == "count":  charbysent[i][j] = "Count Dracula"
        elif charbysent[i][j] == "dracula":  charbysent[i][j] = "Count Dracula"
        elif charbysent[i][j] == "jonathan":  charbysent[i][j] = "Jonathan Harker"
        elif charbysent[i][j] == "mina":  charbysent[i][j] = "Mina Harker"
        elif charbysent[i][j] == "arthur":  charbysent[i][j] = "Arthur Holwood"
        elif charbysent[i][j] == "holwood":  charbysent[i][j] = "Arthur Holwood"
        elif charbysent[i][j] == "quincey":  charbysent[i][j] = "Quincey Morris"
        elif charbysent[i][j] == "morris":  charbysent[i][j] = "Quincey Morris"
        elif charbysent[i][j] == "renfield":  charbysent[i][j] = "Renfield"
        elif charbysent[i][j] == "john":  charbysent[i][j] = "John Seward"
        elif charbysent[i][j] == "seward":  charbysent[i][j] = "John Seward"
        elif charbysent[i][j] == "abraham":  charbysent[i][j] = "Prof. Van Helsing"
        elif charbysent[i][j] == "helsing":  charbysent[i][j] = "Prof. Van Helsing"
        elif charbysent[i][j] == "lucy":  charbysent[i][j] = "Lucy Westenra"
        elif charbysent[i][j] == "westenra":  charbysent[i][j] = "Lucy Westenra"

# Removing double character by sentence

for i in range(0, len(charbysent)):
    charbysent[i] = list(set(charbysent[i]))

# Creating set
    
charset = []
for i in range(0, len(charbysent)):
    sentence = []
    for j in range(0, len(charbysent[i])):
        if (charbysent[i][j] == "Count Dracula"): sentence.append("a")
        elif (charbysent[i][j] == "Jonathan Harker"): sentence.append("b")
        elif (charbysent[i][j] == "Mina Harker"):  sentence.append("c")
        elif (charbysent[i][j] == "Arthur Holwood"): sentence.append("d")
        elif (charbysent[i][j] == "Quincey Morris"): sentence.append("e")
        elif (charbysent[i][j] == "Renfield"):  sentence.append("f")
        elif (charbysent[i][j] == "John Seward"): sentence.append("g")
        elif (charbysent[i][j] == "Prof. Van Helsing"): sentence.append("h")
        elif (charbysent[i][j] == "Lucy Westenra"): sentence.append("i")
    charset.append(sentence)

# A priori algorithm

def apriori(s, min_support, max_length):
    c = Counter()

    for t in s:
        c.update(set(t))

    frequent_sets = {1:[[set(i) for i in c if c[i]>=min_support],[c[i] for i in c if c[i]>=min_support]]}

    current_level = 1

    while len(frequent_sets[current_level][0]) >= 0 and (current_level < max_length):
        frequent_sets[current_level+1] = [[],[]]
        for i, j in combinations(frequent_sets[current_level][0],2):
            new_set = i.copy()
            new_set.update(j)
            if len(new_set) == current_level + 1:
                support = sum([new_set.issubset(set(t)) for t in s])
                if support >= min_support:
                    if new_set not in frequent_sets[current_level+1][0]:
                        frequent_sets[current_level+1][0].append(new_set)
                        frequent_sets[current_level+1][1].append(support)
        current_level += 1
    
    return list(chain(*[[(x,y) for x,y in zip(a[0],a[1])] for a in frequent_sets.values()]))

# Applying a priori
    
fis = apriori(charset, 4, 10)
fislist = []
for i in range(0, len(fis)):
    fislist.append([list(fis[i][0]), fis[i][1]])

# Getting FIS with character names

for i in range(0, len(fislist)):
    for j in range(0, len(fislist[i][0])):
        if (fislist[i][0][j] == "a"): fislist[i][0][j] = "Count Dracula"
        elif (fislist[i][0][j] == "b"): fislist[i][0][j] = "Jonathan Harker"
        elif (fislist[i][0][j] == "c"):  fislist[i][0][j] = "Mina Harker"
        elif (fislist[i][0][j] == "d"): fislist[i][0][j] = "Arthur Holwood"
        elif (fislist[i][0][j] == "e"): fislist[i][0][j] = "Quincey Morris"
        elif (fislist[i][0][j] == "f"):  fislist[i][0][j] = "Renfield"
        elif (fislist[i][0][j] == "g"): fislist[i][0][j] = "John Seward"
        elif (fislist[i][0][j] == "h"): fislist[i][0][j] = "Prof. Van Helsing"
        elif (fislist[i][0][j] == "i"): fislist[i][0][j] = "Lucy Westenra"