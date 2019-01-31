import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import string, re
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
#import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import argparse

# To take files from command line.
parser = argparse.ArgumentParser()
parser.add_argument("file",type = str, nargs='+',help = "Enter 3 file names: .csv, .txt, .csv") 
args = parser.parse_args()

# To read main file consisting of tweets
df = pd.read_csv(args.file[0], delimiter=',')
text=[]
text = list(df['text'])
plane=list(df['airline'])

# DATA CLEANING
text = [''.join(c for c in s if c not in string.punctuation) for s in text]               # To remove punctuation
text = [i.lower() for i in text]                                                          # To convert into lower case
stop = set(stopwords.words('english'))
text = [word_tokenize(i) for i in text]   
tweets = []
text1 = []
for i in text:                                                                            
    i = [w for w in i if not w in stop]                                                   # To remove stop words
    i = [w for w in i if not re.search(r'^-?[0-9]+(.[0-9]+)?$', w)]                       # To remove numbers   
    text1.append(i)
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_token=[]
for sent in text1:                                                                        # Lemmatization to convert tokens to canonical form
    tweets = []
    for token in sent:
        token = wordnet_lemmatizer.lemmatize(token)
        token = wordnet_lemmatizer.lemmatize(token,pos='v')
        tweets.append(token)
    lemmatized_token.append(tweets)
	
##########################################################################
# COMMENTED INTENTIONALLY                                                #
# Code to find top 100 words as per frequency to check for unwanted and  #
# redundant words. Added stop words based on this analysis.              #
########################################################################## 
#from collections import Counter
#count=[]
#for i in lemmatized_token:
#    for j in i:
#        count.append(j)
#count
#count=Counter(word for word in count)
#count.most_common(100)

# To remove added stop words as analysed from above commented code
added_stop_words = ['much','two','youve','ever','since','aa','jfk','dfw','americanairlines','didnt','thats','still','ive','really','virginamerica','usairways','americanair','southwestair','jetblue','airline','\'','\"','plane','flight','flightled','wo','hows', 'unite', 'u', 'get','im','could','would','make','cant','amp','dont','number','one','need','w','theyd','take','even']
lemmatized_token = [[c for c in s if c not in added_stop_words] for s in lemmatized_token]

# MODELLING
bigram = gensim.models.Phrases(lemmatized_token)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
lemmatized_token = [bigram_mod[line] for line in lemmatized_token]  				# bigrams for words appearing together often created  
dictionary = Dictionary(lemmatized_token)                                           # creating dictionary to identify mapping between tokens and their integer ids											
dictionary.filter_extremes(no_below=2)                                              # keep tokens which are contained in atleast in 2 tweets 
corpus = [dictionary.doc2bow(text) for text in lemmatized_token]					# gives number to each unique token along with its counts in corpus

##########################################################################
# COMMENTED INTENTIONALLY                                                #
# Code to choose optimal k. Loop to calculate coherence scores of number #
# of topics from 1 to 20.                                                #
##########################################################################
#limit=21
#start=1 
#step=1
#coherence_values = []
#model_list = []
#for num_topics in range(start, limit, step):
#    model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
#    model_list.append(model)
#    coherencemodel = CoherenceModel(model=model, texts=lemmatized_token, dictionary=dictionary, coherence='c_v')
#    coherence_values.append(coherencemodel.get_coherence())
#coherence_values

##########################################################################
# COMMENTED INTENTIONALLY                                                #
# Graph to plot coherence scores vs number of topics.                    #
# Visualization for number of topics equal to 7							 #
##########################################################################
#limit=21; start=1; step=1;
#x = range(start, limit, step)
#plt.plot(x, coherence_values)
#plt.xlabel("Num Topics")
#plt.ylabel("Coherence score")
#plt.legend(("coherence_values"), loc='best')
#plt.show()																			# graph to plot coherence scores vs number of topics
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(model_list[6], corpus, dictionary)                   # visualization for number of topics equal to 7
#vis

#model_list[6].save("C:/Users/shailaja/Desktop/Shireen/NLP/Final Project/LDAModelPP.txt")         # to chosen lda model

# Topic Names :
# Topic 0 - call_center 
# Topic 1 - waiting
# Topic 2 - cancelling
# Topic 3 - delays
# Topic 4 - customer_service
# Topic 5 - travel
# Topic 6 - thanksstr(p.iloc[i,3]) for i in range(len(p))
a = args.file[1]
lda_model = LdaModel.load(a)														# to load lda model
#print(lda_model.show_topics())

# To write topics obtained to file
topics = open(args.file[2],'w')
topics.write(str(lda_model.show_topics()))