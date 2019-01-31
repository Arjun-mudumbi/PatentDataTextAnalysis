
#%%%%%%%Importing the required packages 
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
#%%%%%%%%%reading in the address csv file
address=pd.read_csv("res.csv",names=[0])
#%%%%%Importing NLTK and required modules
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
#%%%using address csv file to read in all the patent data and dump into a pandas dataframe 
patentData=pd.DataFrame()
for i in range(len(address)):
    data=pd.read_json(address.iloc[i,0])   
    patents=data["content"].iloc[0]
    patents=pd.DataFrame(patents)
    d=patents.docs
    p=[d[i] for i in range(len(d))]
    p=pd.DataFrame(p)
    patentData=patentData.append(p)
p=patentData
p=p.dropna()
patentData=patentData.dropna()
#%%%%%%%%Data cleansing by removing the stop words , lematizing, Removing the punctuations
filtered_sentence=""
p["filtered_sentence"]=""

for i in range(len(p)):
    stem_text=""
    text1=re.sub('\n', '', str(p.ab_en.iloc[i]))
    text1=re.sub('<.*?>', '', text1)
    text="".join([w for w in text1 ])
    word_tokens =  re.split('\W+',text)
    filtered_sentence =" ".join([w for w in word_tokens if not w in stop_words]) 
    text="".join([w for w in str(filtered_sentence) if w not in string.punctuation])
    word_tokens =  re.split('\W+',text)
   
    for w in word_tokens:  
       stem_text=stem_text+ps.lemmatize(w)+" "         
    p["filtered_sentence"].iloc[i]=stem_text
    word2vec_tokenize = word_tokenize(p["filtered_sentence"].iloc[i])
#%%%%%Finding out the bigrams from the filtered text
mystring=p.iloc[i,4]
msystring=mystring.split(" ")
list(nltk.bigrams(msystring))
#%%%%%%Combining all the filtered text
megastring=""
for i in range(len(p)):
    megastring=megastring+str(p.iloc[i,4])+""
#%% Finding the bigram collocation score to find the most important bigrams 
from nltk.collocations import BigramCollocationFinder
def bi(text):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder=BigramCollocationFinder.from_words(word_tokenize(text))
    finder.apply_freq_filter(10)
    finder.nbest(bigram_measures.pmi, 5) 
    return finder.ngram_fd.items()
mybigram=bi(megastring)
#%%%%%555
mybigram
mybigram=list(mybigram)
#%%%%%%%creating the list of bigrams
mybigramlist=[]
for j in range(len(mybigram)):
    word=mybigram[j][0]
    mybigramlist.append(word[0]+" "+word[1])
#%%% generating the list of bigrams 0
len(mybigramlist)
mybigramlist=set(mybigramlist)
mybigramlist=list(mybigramlist)
#%%%%%%%%%Generating the TF-IDF matrix using the bigrams as the vocabulary
vectorizer = TfidfVectorizer(vocabulary=mybigramlist,ngram_range=(2,2))
dtm2 = vectorizer.fit_transform(str(p.iloc[i,4]) for i in range(len(p)))
cols=pd.DataFrame(dtm2.toarray(),columns=vectorizer.get_feature_names())
vectorizer.get_feature_names()
#%%%%%%% Extracting the year and months of the patent publication
publication=[str(p.pd.iloc[i]) for i in range(len(p))]
publication_year=[]
publication_month=[]
for i in range(len(p)):
    newstring=publication[i]
    publication_year.append(int(newstring[:4]))
    publication_month.append(int(newstring[4:6]))
#%%%%Dictionary to create the months
subs={1:"January",
      2:"February",
      3:"March",
      4:"April",
      5:"May",
      6:"June",
      7:"July",
      8:"August",
      9:"September",
      10:"October",
      11:"November",
      12:"December"}
publicationmonths=[subs.get(item,item)  for item in publication_month]
#%%%%creating the publiaction year and publication month list
publication_year=pd.Series(publication_year)
cols["publication_year"]=publication_year
publication_month=pd.Series(publicationmonths)
cols["publication_month"]=publication_month
#%%%%%%%%using the polaritydataframe csv file to generate 
year=pd.Series()
month=pd.Series()
polarity=pd.read_csv("polaritydataframe.csv")
mymonths=['January','February','March','April','May','June','July','August','September','October','November','December']
polarity["month"]=pd.Categorical(polarity['month'], categories=mymonths, ordered=True)
polarity=polarity.sort_values(["year","month"])
vals=polarity["values"].shift(-1)
vals=vals.fillna(0)
polarity2=polarity
polarity2["values"]=vals
#%%%%%%%%polarity with two month shift
vals=polarity["values"].shift(-2)
vals=vals.fillna(0)
polarity3=polarity
polarity3["values"]=vals
#%%%%%%%%polarity with 3 months shift
vals=polarity["values"].shift(-3)
vals=vals.fillna(0)
polarity4=polarity
polarity4["values"]=vals
#%%%%%%polarity with 4 months shift
val=polarity["values"].shift(-4)
vals=vals.fillna(0)
polarity5=polarity
polarity5["values"]=vals
#%%%%%%%polarity with 5 months shift
vals=polarity["values"].shift(-5)
vals=vals.fillna(0)
polarity6=polarity
polarity6["values"]=vals
#%%%%%%%%%%polarity with 6 months shift
vals=polarity["values"].shift(-6)
vals=vals.fillna(0)
polarity7=polarity
polarity["values"]=vals
#%%%%%%%polarity dataframe generation
polarity_dict={}
for i in range(len(polarity)):
    polarity_dict[polarity7["year"].iloc[i], polarity7["month"].iloc[i]]=polarity7["values"].iloc[i]
polarity=polarity_dict
#%%%%%%%%generating sentiments using polarity dataframe 
sentiments=[]
for i in range(len(p)):
    print(polarity[publication_year[i],publication_month[i]])
    sentiments.append(polarity[(publication_year[i], publication_month[i])])
#%%%%%%%appending the sentiments to the cols
sentiments=pd.Series(sentiments)
sentiments=sentiments.fillna(0)
cols["sentiments"]=sentiments
#%%%%creating the target variable
poscols=cols
pos=poscols.sentiments>=0
pos=[int(i) for i in pos]
pos=pd.Series(pos)
#%%create the matrix of predictors and the target variable
x=poscols.drop(["publication_year","publication_month","sentiments"],axis=1)
y=pos

#%%%%%%Splitting into train and test matrix
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain ,ytest= train_test_split(x,y,test_size=0.2)
#%%%%Apply Smote to resample the undersampled 
sm = SMOTE(random_state=12)
xtrain, ytrain = sm.fit_sample(xtrain, ytrain)

#%%%%%%%Implementing the Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(solver="newton-cg",multi_class="multinomial")
classifier.fit(xtrain, ytrain)
ypred=classifier.predict(xtest)
#%%%%% Implementing the ROC and AUC curve 
from sklearn.metrics import roc_curve,auc
y_predict_probabilities = classifier.predict_proba(xtest)[:,1]
fpr, tpr, _ = roc_curve(ytest, y_predict_probabilities)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
#%%%%%%%%%%%%
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
wordcloud = WordCloud().generate(megastring)

#display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
