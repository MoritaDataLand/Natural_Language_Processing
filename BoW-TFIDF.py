# -*- coding: utf-8 -*-
"""
@channel Morita DataLand
@author Morita Tarvirdians
@email tarvirdians.morita@gmail.com
@desc Bag of Words and TF-IDF tutorial

"""
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

text = """A major drawback of statistical methods is that they require elaborate feature engineering. 
Since the early 2010s, the field has thus largely abandoned statistical methods and shifted to neural networks for machine learning. 
Popular techniques include the use of word embeddings to capture semantic properties of words, and an increase in end-to-end learning of a higher-level task (e.g., question answering) instead of relying on a pipeline of separate intermediate tasks (e.g., part-of-speech tagging and dependency parsing).
In some areas, this shift has entailed substantial changes in how NLP systems are designed, such that deep neural network-based approaches may be viewed as a new paradigm distinct from statistical natural language processing. 
For instance, the term neural machine translation (NMT) emphasizes the fact that deep learning-based approaches to machine translation directly learn sequence-to-sequence transformations, obviating the need for intermediate steps such as word alignment and language modeling that was used in statistical machine translation (SMT). 
Latest works tend to use non-technical structure of a given task to build proper neural network
"""

#cleaning text
sentences = sent_tokenize(text)
stemmer = PorterStemmer()
corpus = []

for sent in sentences:
    review = re.sub("[^a-zA-Z]", " ", sent)
    review = re.sub("\b[a-zA-Z]\b", " ", review)
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)

#vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
bow = cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
tfidf = tf.fit_transform(corpus).toarray()




