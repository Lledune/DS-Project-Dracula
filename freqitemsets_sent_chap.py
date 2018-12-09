from collections import Counter
from itertools import combinations, chain
import re
import nltk
from nltk.corpus import stopwords
import csv

# Importing data by chapters

filepath = "D:/Google Drive/DATS2MS/LINMA2472 Algorithms in data science/Project/Dracula/dracula.txt"

startStops = [("CHAPTER I", "CHAPTER II"), ("CHAPTER II", "CHAPTER III"),
              ("CHAPTER III", "CHAPTER IV"), ("CHAPTER IV", "CHAPTER V"),
              ("CHAPTER V", "CHAPTER VI"), ("CHAPTER VI", "CHAPTER VII"),
              ("CHAPTER VII", "CHAPTER VIII"), ("CHAPTER VIII", "CHAPTER IX"),
              ("CHAPTER IX", "CHAPTER X"), ("CHAPTER X", "CHAPTER XI"),
              ("CHAPTER XI", "CHAPTER XII"), ("CHAPTER XII", "CHAPTER XIII"),
              ("CHAPTER XIII", "CHAPTER XIV"), ("CHAPTER XIV", "CHAPTER XV"),
              ("CHAPTER XV", "CHAPTER XVI"), ("CHAPTER XVI", "CHAPTER XVII"),
              ("CHAPTER XVII", "CHAPTER XVIII"), ("CHAPTER XVIII", "CHAPTER XIX"),
              ("CHAPTER XIX", "CHAPTER XX"), ("CHAPTER XX", "CHAPTER XXI"),
              ("CHAPTER XXI", "CHAPTER XXII"), ("CHAPTER XXII", "CHAPTER XXIII"),
              ("CHAPTER XXIII", "CHAPTER XXIV"), ("CHAPTER XXIV", "CHAPTER XXV"),
              ("CHAPTER XXV", "CHAPTER XXVI"),("CHAPTER XXVI", "CHAPTER XXVII"),
              ("CHAPTER XXVII", "THE END")]

chapters = []

with open(filepath, 'r') as myfile:
    data=myfile.read().replace('\n', ' ')
del myfile

for i in range(0, len(startStops)):
    Start = startStops[i][0]
    End = startStops[i][1]
    expression = re.compile(r'%s.*?%s' % (Start,End), re.S)
    chapters.append(expression.search(data).group(0))
del data, startStops, End, Start
    
# Creating sentences
    
sentencesc = []
for i in range(0, len(chapters)):
    sentencesc.append(nltk.tokenize.sent_tokenize(chapters[i]))

# Stopwords
    
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# Basic text preprocessing
for k in range(0, len(sentencesc)):
    for i in range(0, len(sentencesc[k])):
        sentence = sentencesc[k][i]
        cleanr = re.compile('<.*?>')
        sentence = sentence.lower()
        sentence = re.sub(cleanr, ' ', sentence)
        sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        sentence = re.sub(r'[.|,|)|(|\|/|_|-|;]',r' ',sentence)
        sentencesc[k][i] = sentence
del sentence

wordssentc = []
for k in range(0, len(sentencesc)):
    wordssent = []
    for i in range(0, len(sentencesc[k])):
        sentence = sentencesc[k][i]
        words = [word for word in sentence.split() if word not in stop]
        wordssent.append(words)
    wordssentc.append(wordssent)
del sentence, wordssent
    
# Defining characters

charlist = ["count", "dracula", "jonathan", "mina", "murray","arthur", "holwood",
             "quincey", "morris", "renfield", "john", "seward", "abraham",
             "helsing", "lucy", "westenra"]

# Keeping characters

charbysentc = []
for k in range(0, len(wordssentc)):
    charbysent = []
    for i in range(0, len(wordssentc[k])):
        charinsent = []
        for j in range(0, len(wordssentc[k][i])):
            if (wordssentc[k][i][j] in charlist):
                charinsent.append(wordssentc[k][i][j])
            if charinsent == []: continue
            charbysent.append(charinsent)
    charbysentc.append(charbysent)
del charbysent, charinsent

# Replacing the names of characters

for k in range(0, len(charbysentc)):
    for i in range(0, len(charbysentc[k])):
        for j in range(0, len(charbysentc[k][i])):
                    if charbysentc[k][i][j] == "count":  charbysentc[k][i][j] = "Count Dracula"
                    elif charbysentc[k][i][j] == "dracula":  charbysentc[k][i][j] = "Count Dracula"
                    elif charbysentc[k][i][j] == "jonathan":  charbysentc[k][i][j] = "Jonathan Harker"
                    elif charbysentc[k][i][j] == "mina":  charbysentc[k][i][j] = "Mina Murray"
                    elif charbysentc[k][i][j] == "murray":  charbysentc[k][i][j] = "Mina Murray"
                    elif charbysentc[k][i][j] == "arthur":  charbysentc[k][i][j] = "Arthur Holwood"
                    elif charbysentc[k][i][j] == "holwood":  charbysentc[k][i][j] = "Arthur Holwood"
                    elif charbysentc[k][i][j] == "quincey":  charbysentc[k][i][j] = "Quincey Morris"
                    elif charbysentc[k][i][j] == "morris":  charbysentc[k][i][j] = "Quincey Morris"
                    elif charbysentc[k][i][j] == "renfield":  charbysentc[k][i][j] = "Renfield"
                    elif charbysentc[k][i][j] == "john":  charbysentc[k][i][j] = "Dr. Seward"
                    elif charbysentc[k][i][j] == "seward":  charbysentc[k][i][j] = "Dr. Seward"
                    elif charbysentc[k][i][j] == "abraham":  charbysentc[k][i][j] = "Prof. Van Helsing"
                    elif charbysentc[k][i][j] == "helsing":  charbysentc[k][i][j] = "Prof. Van Helsing"
                    elif charbysentc[k][i][j] == "lucy":  charbysentc[k][i][j] = "Lucy Westenra"
                    elif charbysentc[k][i][j] == "westenra":  charbysentc[k][i][j] = "Lucy Westenra"

# Removing double character by sentence

for k in range(0, len(charbysentc)):
    for i in range(0, len(charbysentc[k])):
        charbysentc[k][i] = list(set(charbysentc[k][i]))

# Creating set

charsetc = []
for k in range(0, len(charbysentc)):  
    charset = []
    for i in range(0, len(charbysentc[k])):
        sentence = []
        for j in range(0, len(charbysentc[k][i])):
            if (charbysentc[k][i][j] == "Count Dracula"): sentence.append("a")
            elif (charbysentc[k][i][j] == "Jonathan Harker"): sentence.append("b")
            elif (charbysentc[k][i][j] == "Mina Murray"):  sentence.append("c")
            elif (charbysentc[k][i][j] == "Arthur Holwood"): sentence.append("d")
            elif (charbysentc[k][i][j] == "Quincey Morris"): sentence.append("e")
            elif (charbysentc[k][i][j] == "Renfield"):  sentence.append("f")
            elif (charbysentc[k][i][j] == "Dr. Seward"): sentence.append("g")
            elif (charbysentc[k][i][j] == "Prof. Van Helsing"): sentence.append("h")
            elif (charbysentc[k][i][j] == "Lucy Westenra"): sentence.append("i")
        charset.append(sentence)
    charsetc.append(charset)
del charset, sentence

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

# Defining the application of a priori on the chapters, returning list of lists

def apply_apriori (chap, min_support, max_length):
    fisc = []
    for k in range(0, len(chap)):
        fis = apriori(chap[k], min_support, max_length)
        fislist = []
        for i in range(0, len(fis)):
            fislist.append([list(fis[i][0]), fis[i][1]])
        fisc.append(fislist)
    for k in range(0, len(fisc)):
        for i in range(0, len(fisc[k])):
            for j in range(0, len(fisc[k][i][0])):
                if (fisc[k][i][0][j] == "a"): fisc[k][i][0][j] = "Count Dracula"
                elif (fisc[k][i][0][j] == "b"): fisc[k][i][0][j] = "Jonathan Harker"
                elif (fisc[k][i][0][j] == "c"):  fisc[k][i][0][j] = "Mina Murray"
                elif (fisc[k][i][0][j] == "d"): fisc[k][i][0][j] = "Arthur Holwood"
                elif (fisc[k][i][0][j] == "e"): fisc[k][i][0][j] = "Quincey Morris"
                elif (fisc[k][i][0][j] == "f"):  fisc[k][i][0][j] = "Renfield"
                elif (fisc[k][i][0][j] == "g"): fisc[k][i][0][j] = "Dr. Seward"
                elif (fisc[k][i][0][j] == "h"): fisc[k][i][0][j] = "Prof. Van Helsing"
                elif (fisc[k][i][0][j] == "i"): fisc[k][i][0][j] = "Lucy Westenra"
    return(fisc)

# Applying apriori by chapter
    
fisc = apply_apriori(charsetc, 5, 10)
            
# Getting list of frequent character sets

freqchard = []
for k in range(0, len(fisc)):
    for i in range(0, len(fisc[k])):
        freqchard.append(fisc[k][i][0])

freqchar = []
for i in range(0, len(freqchard)):
    if freqchard[i] not in freqchar: freqchar.append(freqchard[i])
del freqchard

# Exporting in csv

for i in range(0, len(fisc)):
    writer = csv.writer(open("fisc"+str(i)+".csv", "w"), dialect='excel')
    writer.writerows(fisc[i])

writer = csv.writer(open("freqitemsets.csv", "w"), dialect='excel')
writer.writerows(fisc)

writer = csv.writer(open("fuckoff.csv", "w"))

