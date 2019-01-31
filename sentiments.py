#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55

import numpy as np
import pandas as pd
import re
from textblob import TextBlob
#%%%%%%%%%%%
ad=["target-address"]
address=pd.read_csv("Address.csv",names=ad)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
articles = []
fields = ['HD', 'CR', 'WC', 'PD', 'ET', 'SN', 'SC', 'ED', 'PG', 'LA', 'CY', 'LP',
              'TD', 'CT', 'RF', 'CO', 'IN', 'NS', 'RE', 'IPC', 'IPD', 'PUB', 'AN']
factiva=pd.DataFrame(columns=fields) 
#%%%%%555
for d in range(len(address)):
    with open(address["target-address"].iloc[d], 'r') as infile:
        data = infile.read()
    start = re.search(r'\n HD\n', data).start()
    for m in re.finditer(r'Document [a-zA-Z0-9]{25}\n', data):
        end = m.end()
        a = data[start:end].strip()
        a = '\n   ' + a
        articles.append(a)
        start = end
    mydata=[]
    for a in articles:
        used = [f for f in fields if re.search(r'\n   ' + f + r'\n', a)]
        unused = [[i, f] for i, f in enumerate(fields) if not re.search(r'\n   ' + f + r'\n', a)]
        fields_pos = []
        for f in used:
            f_m = re.search(r'\n   ' + f + r'\n', a)
            f_pos = [f, f_m.start(), f_m.end()]
            fields_pos.append(f_pos)
        obs = []
        n = len(used)
        for i in range(0, n):
            used_f = fields_pos[i][0]
            start = fields_pos[i][2]
            if i < n - 1:
                end = fields_pos[i + 1][1]
            else:
                end = len(a)
            content = a[start:end].strip()
            obs.append(content)
        for f in unused:
            obs.insert(f[0], '')
        mydata.append(pd.DataFrame(np.array(obs).reshape(1,23),columns=fields))
for i in range(len(mydata)):
    factiva=factiva.append(mydata[i])
#%%%%%%    
factiva.to_csv("Factiva.csv")
#%%%%%%%%%%%%%%%%%%%%%%%%%
factiva["sentiments"]=""
for i in range(len(factiva)):
    blob=TextBlob(factiva["LP"].iloc[i])
    factiva["sentiments"].iloc[i]=blob.sentiment.polarity
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
factiva['HD'].replace('', np.nan, inplace=True)
factiva.dropna(inplace=True)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
dates=factiva["PD"]
dates=dates.str.split(" ", expand=True)
factiva["publication year"]=(dates.iloc[:,2])
#%%%%%%%%%%%%5
polarity={}
for i in range(1989,2019):
    polarity[i]=factiva.loc[(factiva['publication year']==str(i)),'sentiments'].mean()
#%%%%%%%


#%%%%%%%%%%%%%%%55
