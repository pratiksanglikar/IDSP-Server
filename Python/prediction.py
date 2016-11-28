import pandas as pd
from sklearn import tree
df = pd.read_csv("D:/msse/2nd_sem/281_KongLi/Project-2/database/newnewtraindata1.csv") 
#dummy_disease = pd.get_dummies(df['disease'], prefix = 'disease')
#print dummy_disease.head
#col_to_keep = ['cases', 'location']
#data = df[col_to_keep].join(dummy_disease.ix[: 'disease_Arsenicosis':])
#print data.head()
#data['intercept'] = 1.0
import statsmodels.api as sm
train_cols = ['location','disease','ycases']
logit = sm.Logit(df['cases'], df[train_cols])
result = logit.fit()
print result.summary()

print result.conf_int()

import numpy as np
print np.exp(result.params)

params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)

print result.predict([48,1,0])