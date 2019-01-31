
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF 
from sklearn.decomposition import FastICA
from nltk.tokenize import word_tokenize 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
address=pd.read_csv("res.csv",names=[0])
#%%%%%%%%%%%%%%
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
set(stopwords.words('english'))
#%%%%%%%%5
patentData=pd.DataFrame()
for i in range(len(address)):
    data=pd.read_json(address.iloc[i,0])
    patents=data.iloc[0,0]
    patents=pd.DataFrame(patents)
    d=patents.iloc[:,0]
    p=[d[i] for i in range(len(d))]
    p=pd.DataFrame(p)
    patentData=patentData.append(p)
p=patentData
#%%%%%%%%%%%%%%%%%%%%5555
#vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
#dtm = vectorizer.fit_transform(str(p.iloc[i,0]) for i in range(len(p)))
#cols=pd.DataFrame(dtm.toarray(),columns=vectorizer.get_feature_names())
#vectorizer.get_feature_names()
#lsa = TruncatedSVD(5, algorithm = 'arpack')
#dtm=dtm.asfptype()
#dtm_lsa = lsa.fit_transform(dtm)
#dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
#%%%%%%%%%%%%555
for i in range(len(p)):
    word_tokens = word_tokenize(p.iloc[i,0]) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
            print(filtered_sentence)
#%%%%%%%%%%%%Implementing TfIDF vectorizer
vectorizer = TfidfVectorizer()
dtm = vectorizer.fit_transform(str(p.iloc[i,0]) for i in range(len(p)))
cols=pd.DataFrame(dtm.toarray(),columns=vectorizer.get_feature_names())
vectorizer.get_feature_names()
#%%%%%%%%%%%%%Implementing SVD
lsa = TruncatedSVD(100, algorithm = 'arpack')
dtm=dtm.asfptype()
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
#%%%%%%%%55
dtm_lsa=pd.DataFrame(dtm_lsa)
dtm_lsa.to_csv("Svd_Results.csv")
#%%%%%%%%%%%%Screeplot
var1=np.cumsum(np.round(lsa.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.xlabel("n_components")
plt.ylabel("variance explained")
#%%%%%%%%%%%%%%%%Implementing Non negative matrix featurization
nmf=NMF(n_components =100)
dtm_nmf=nmf.fit_transform(dtm)
dtm_nmf = Normalizer(copy=False).fit_transform(dtm_nmf)
#%%%%%%%%55
#%%%%%%%%%%%%%%%%%Implementing PCA
pca= PCA(n_components=100)
dtm_pca=pca.fit_transform(dtm.toarray())
dtm_pca = Normalizer(copy=False).fit_transform(dtm_pca)
#%%%%%%%%%%%%%%%555
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.xlabel("n_components")
plt.ylabel("variance explained")
#%%%%%%%%%%%%Implementing FAST ICA
fica=FastICA(n_components=100)
dtm_fica=fica.fit_transform(dtm.toarray())
dtm_fica=Normalizer(copy=False).fit_transform(dtm_fica)
#%%%%%%%%%%%%%%%%%%%Implementing LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=100)
dtm_lda=lda.fit_transform(dtm)
dtm_lda=Normalizer(copy=False).fit_transform(dtm_lda)
#%%%%%%%%%%%%55implementing MDS
from sklearn.manifold import MDS
mds=MDS(n_components=100)
dtm_mds=mds.fit_transform(dtm.toarray())
dtm_mds=Normalizer(copy=False).fit_transform(dtm_mds)
#%%%%implementing ISOMAP
from sklearn.manifold import Isomap
ism = Isomap(n_components=100)
dtm_ism=ism.fit_transform(dtm.toarray())
dtm_ism=Normalizer(copy=False).fit_transform(dtm_ism)
#%%%%%%%%%%%%%%5 implementing Laplacian EigenMap
from sklearn.manifold import SpectralEmbedding
lle= SpectralEmbedding(n_components=100)
dtm_lle=lle.fit_transform(dtm.toarray())
dtm_lle=Normalizer(copy=False).fit_transform(dtm_lle)
#%%%%%%%%%%%%%%%%555 implementing TSNE
from sklearn.manifold import TSNE
tsne=TSNE(n_components=100,method='exact')
dtm_tsne=tsne.fit_transform(dtm.toarray())
dtm_tsne=Normalizer(copy=False).fit_transform(dtm_tsne)


