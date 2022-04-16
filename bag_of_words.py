import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

def bow():

    try:
        text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
        ])
        text_clf.fit(twenty_train.data, twenty_train.target)
        predicted = text_clf.predict(twenty_test.data)
        np.mean(predicted == twenty_test.target)
    except:
        print("Bag of words ne fonctionne pas")

"""Let's begin with covering the first one, using a bag-of-words approach.
Bag-of-words
We'll build a pipeline of counting words and reweighing them according to their frequency. The final classifier is a
random forest. We train this on our training dataset:
"""