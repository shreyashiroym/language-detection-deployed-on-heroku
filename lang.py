#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")


data=pd.read_csv('Language Detection.csv')
data.head()

data['Language'].value_counts()

x=data['Text']
y=data['Language']

import re
lang = []
for text in x:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    lang.append(text)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(lang).toarray()

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

con=pd.crosstab(y_pred,y_test)

plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True, cmap="Blues")
plt.show()

for i in range(con.shape[0]):
    TP=con.iloc[i,i]
    FP=con.iloc[i,:].sum()-TP
    FN=con.iloc[:,i].sum()-TP
    TN=con.sum().sum()-TP-FP-FN
    Accuracy=(TP+TN)/con.sum().sum()
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F1_score=(2*Precision*Recall)/(Precision+Recall)
    print(con.index[i],Accuracy, Precision,Recall, F1_score)

from sklearn.metrics import classification_report
classification_report(y_pred,y_test)

pd.DataFrame(classification_report(y_pred,y_test,output_dict=True)).T

import pickle
pickle.dump(classifier, open("lang.pkl","wb"))