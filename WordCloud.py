
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF 
from sklearn.decomposition import FastICA
from nltk.tokenize import word_tokenize 
import re
from imblearn.over_sampling import SMOTE
#%%%%%%%%%%%%%%%%%
address=pd.read_csv("res.csv",names=[0])
#%%%%%%%%%%%%%%%%5
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet  
from nltk.stem import WordNetLemmatizer

ps = WordNetLemmatizer()
stop_words=set(stopwords.words('english'))
stop_words=list(stop_words)
stop_words.append('n')
stop_words=set(stop_words)
#%%%%%%%%%%%%%%%%5555
patentData=pd.DataFrame()
for i in range(len(address)):
    data=pd.read_json(address.iloc[i,0])   
    patents=data["content"].iloc[0]
#    print(address.iloc[i,0])
#    print(patents)
    patents=pd.DataFrame(patents)
    d=patents.docs
    p=[d[i] for i in range(len(d))]
    p=pd.DataFrame(p)
    patentData=patentData.append(p)
p=patentData
p=p.dropna()
patentData=patentData.dropna()
#%%%%%%%%%%%%%%%%%%%%5
filtered_sentence=""
p["filtered_sentence"]=""

for i in range(len(p)):
    #print(i)
    stem_text=""
    text1=re.sub('\n', '', str(p.ab_en.iloc[i]))
    text1=re.sub('<.*?>', '', text1)
    text="".join([w for w in text1 ])
    word_tokens =  re.split('\W+',text)
    #print(text)
    filtered_sentence =" ".join([w for w in word_tokens if not w in stop_words]) 
    text="".join([w for w in str(filtered_sentence) if w not in string.punctuation])
    word_tokens =  re.split('\W+',text)
   
    for w in word_tokens:  
        #print(ps.stem(w))
       #stem_text=stem_text.join([ps.stem(w)])
       stem_text=stem_text+ps.lemmatize(w)+" "
         
    p["filtered_sentence"].iloc[i]=stem_text
    word2vec_tokenize = word_tokenize(p["filtered_sentence"].iloc[i])
#%%%%%%%%%%%%%%%%%5
mystring=p.iloc[i,4]
msystring=mystring.split(" ")
list(nltk.bigrams(msystring))
#%%%%%%%%%%%55
megastring=""
for i in range(len(p)):
    megastring=megastring+str(p.iloc[i,4])+""
#%%%%%%%
import numpy as np
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from matplotlib import pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import wordcloud
import gensim
from os import path
from PIL import Image

#import gensim packages
from gensim import corpora, models, similarities
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#%%%%%%%%
"augmented reality" in megastring
megastring=megastring.replace("augmented reality","")
megastring=megastring.replace("least one","")
megastring=megastring.replace("based","")
#%%%%%%%5
wordcloud = WordCloud().generate(megastring)

#display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
