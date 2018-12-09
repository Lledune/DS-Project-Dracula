from collections import Counter
from itertools import combinations, chain
import re
import nltk
from nltk.corpus import stopwords

# Importing data

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

for i in range(0, 27):
    Start = startStops[i][0]
    End = startStops[i][1]
    expression = re.compile(r'%s.*?%s' % (Start,End), re.S)
    chapters.append(expression.search(data).group(0))

# Stopwords
    
nltk.download('stopwords')   
stop = set(stopwords.words('english'))

# Basic text preprocessing

for i in range(0, 27):
    chapterUnique = chapters[i]
    cleanr = re.compile('<.*?>')
    chapterUnique = chapterUnique.lower()
    chapterUnique = re.sub(cleanr, ' ', chapterUnique)
    chapterUnique = re.sub(r'[?|!|\'|"|#]',r'',chapterUnique)
    chapterUnique = re.sub(r'[.|,|)|(|\|/|_|-|;]',r' ',chapterUnique)
    chapters[i] = chapterUnique

wordschap = []
for i in range(0, 27):
    chapterUnique = chapters[i]
    words = [word for word in chapterUnique.split() if word not in stop]
    wordschap.append(words)
    
# Defining characters

charlist = ["count", "dracula", "jonathan", "mina", "arthur", "holwood",
             "quincey", "morris", "renfield", "john", "seward", "abraham",
             "helsing", "lucy", "westenra"]

# Keeping characters

charbychap = []
for i in range(0, len(wordschap)):
    charinchap = []
    for j in range(0, len(wordschap[i])):
        if (wordschap[i][j] in charlist):
            charinchap.append(wordschap[i][j])
    charbychap.append(charinchap)

# Replacing the names of characters
    
for i in range(0, len(charbychap)):
    for j in range(0, len(charbychap[i])):
        if (charbychap[i][j] == "count"):  charbychap[i][j] = "Count Dracula"
        elif (charbychap[i][j] == "dracula"):  charbychap[i][j] = "Count Dracula"
        elif (charbychap[i][j] == "jonathan"):  charbychap[i][j] = "Jonathan Harker"
        elif (charbychap[i][j] == "mina"):  charbychap[i][j] = "Mina Harker"
        elif (charbychap[i][j] == "arthur"):  charbychap[i][j] = "Arthur Holwood"
        elif (charbychap[i][j] == "holwood"):  charbychap[i][j] = "Arthur Holwood"
        elif (charbychap[i][j] == "quincey"):  charbychap[i][j] = "Quincey Morris"
        elif (charbychap[i][j] == "morris"):  charbychap[i][j] = "Quincey Morris"
        elif (charbychap[i][j] == "renfield"):  charbychap[i][j] = "Renfield"
        elif (charbychap[i][j] == "john"):  charbychap[i][j] = "John Seward"
        elif (charbychap[i][j] == "seward"):  charbychap[i][j] = "John Seward"
        elif (charbychap[i][j] == "abraham"):  charbychap[i][j] = "Prof. Van Helsing"
        elif (charbychap[i][j] == "helsing"):  charbychap[i][j] = "Prof. Van Helsing"
        elif (charbychap[i][j] == "lucy"):  charbychap[i][j] = "Lucy Westenra"
        elif (charbychap[i][j] == "westenra"):  charbychap[i][j] = "Lucy Westenra"

# A priori algorithm

def apriori(s, min_support, max_length):
    """
    :type s: list
    :param s: list of lists, each sublist is a transaction, each element of the transaction is an item
    :type min_support: int
    :param min_support: minimal number of occurences for an itemset to be frequent
    :type max_length: int
    :param max_length: maximal length of searched itemsets
    :rtype: list
    :return: list of tuples. Each tuple contains 2 elements: the first element is a set representing a frequent
    itemset, the second element is its support in s.
    """

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
    
freqitsets = apriori(charbychap, 2, 3)
