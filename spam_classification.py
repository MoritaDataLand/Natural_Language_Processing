# -*- coding: utf-8 -*-
"""
@channel Morita DataLand
@author Morita Tarvirdians
@email tarvirdians.morita@gmail.com
@desc simple text categorization project for NLP tutorial

"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# read dataset (csv -> pandas dataframe)
df = pd.read_csv("spam_text_message_data.csv")
print(df.head())

df["Category"].replace({'ham': 0 ,'spam': 1}, inplace=True)
print(df.head(10))


# gain insight from data
data = {'category': ['spam', 'ham'],
        'number': [len(df.loc[df.Category==1]), len(df.loc[df.Category==0])]
        }  
df_count = pd.DataFrame(data,columns=['category', 'number'])
print (df_count)

df_count.plot(x ='category', y='number', kind = 'bar')
plt.show()


# cleaning dataset
stemmer = PorterStemmer()
corpus = []

for w in range(len(df['Message'])):
    msg = df['Message'][w]
    msg = re.sub("[^a-zA-Z]", " ", msg)
    msg = msg.lower()
    msg = msg.split()
    msg = [stemmer.stem(word) for word in msg if not word in set(stopwords.words('english'))]
    msg = " ".join(msg)
    corpus.append(msg)



# create word vector
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
tf.fit(corpus)
# print(tf.vocabulary_)
X = tf.transform(corpus).toarray()

Y = df['Category']

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# train model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)

y_pred = model.predict(X_test)

# compute metrics
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)
print(confusion_m)

from sklearn.metrics import accuracy_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)


print("acc", acc, "\n")
print("prec", prec, "\n")
print("rec", rec, "\n")
