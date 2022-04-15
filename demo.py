from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(
 subset='train',
 categories=categories,
 shuffle=True,
 random_state=42
 )
twenty_test = fetch_20newsgroups(
 subset='test',
 categories=categories,
 shuffle=True,
 random_state=42
)

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier()),
])
text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    'wiki.en.vec',
 binary=False, encoding='utf8'
)

import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print("hello")


