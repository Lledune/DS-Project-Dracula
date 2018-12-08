import numpy as np
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud, STOPWORDS

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






