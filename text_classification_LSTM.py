"""
@channel Morita DataLand
@author Morita Tarvirdians
@email tarvirdians.morita@gmail.com
@desc text classification using LSTM

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("spam_text_message_data.csv")
print(df.head())


sns.countplot(x=df["Category"])
plt.show()

X = df["Message"]
Y = df["Category"]

# print(Y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
# print(Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from tensorflow.keras.preprocessing.text import Tokenizer
max_words = 500
tkn = Tokenizer(num_words=max_words)
tkn.fit_on_texts(X_train)
seq = tkn.texts_to_sequences(X_train)
print(seq)

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 100
padded_docs = pad_sequences(seq, padding = 'pre', maxlen= max_len)
print(padded_docs)


from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
embedding_vector_features = 30
model = Sequential()
model.add(Embedding(max_words, embedding_vector_features, input_length=max_len))
model.add(LSTM(256))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit(padded_docs,Y_train,batch_size=128,epochs=10, validation_split=0.2)

text_seq = tkn.texts_to_sequences(X_test)
test_padded = pad_sequences(text_seq, maxlen = max_len)

accr = model.evaluate(test_padded, Y_test)
print(accr)