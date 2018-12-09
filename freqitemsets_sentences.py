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
        if (charbysent[i][j] == "count"):  charbysent[i][j] = "Count Dracula"
        elif (charbysent[i][j] == "dracula"):  charbysent[i][j] = "Count Dracula"
        elif (charbysent[i][j] == "jonathan"):  charbysent[i][j] = "Jonathan Harker"
        elif (charbysent[i][j] == "mina"):  charbysent[i][j] = "Mina Harker"
        elif (charbysent[i][j] == "arthur"):  charbysent[i][j] = "Arthur Holwood"
        elif (charbysent[i][j] == "holwood"):  charbysent[i][j] = "Arthur Holwood"
        elif (charbysent[i][j] == "quincey"):  charbysent[i][j] = "Quincey Morris"
        elif (charbysent[i][j] == "morris"):  charbysent[i][j] = "Quincey Morris"
        elif (charbysent[i][j] == "renfield"):  charbysent[i][j] = "Renfield"
        elif (charbysent[i][j] == "john"):  charbysent[i][j] = "John Seward"
        elif (charbysent[i][j] == "seward"):  charbysent[i][j] = "John Seward"
        elif (charbysent[i][j] == "abraham"):  charbysent[i][j] = "Prof. Van Helsing"
        elif (charbysent[i][j] == "helsing"):  charbysent[i][j] = "Prof. Van Helsing"
        elif (charbysent[i][j] == "lucy"):  charbysent[i][j] = "Lucy Westenra"
        elif (charbysent[i][j] == "westenra"):  charbysent[i][j] = "Lucy Westenra"
        
# Removing double character by sentence

for i in range(0, len(charbysent)):
    charbysent[i] = list(set(charbysent[i]))

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
    
fis = apriori(charbysent, 2, 10)