from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from gensim.models import KeyedVectors
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import regularizers
from gensim.utils import tokenize
from gensim.test.utils import datapath
from gensim.models import FastText
from tensorflow.keras.layers import Embedding
#from eywa.nlu import Classifier
#from eywa.nlu import EntityExtractor
#from pyowm import OWM
#from eywa.nlu import Pattern
import datetime

##viariables
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


text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier()),
])
text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)


model = KeyedVectors.load_word2vec_format(
    'wiki.en.vec',
 binary=False, encoding='utf8'
)


def embed_text(text: str):
 vector_list = [
   model.wv[w].reshape(-1, 1) for w in text_to_word_sequence(text)
   if w in model.wv
 ]
 if len(vector_list) > 0:
   return np.mean(
        np.concatenate(vector_list, axis=1),
        axis=1
   ).reshape(1, 300)
 else:
  return np.zeros(shape=(1, 300))

assert embed_text('training run').shape == (1, 300)

train_transformed = np.concatenate(
 [embed_text(t) for t in twenty_train.data]
)
rf = RandomForestClassifier().fit(train_transformed, twenty_train.target)

test_transformed = np.concatenate(
 [embed_text(t) for t in twenty_test.data]
)
predicted = rf.predict(test_transformed)
np.mean(predicted == twenty_test.target)

embedding = layers.Embedding(
 input_dim=5000, 
 output_dim=50, 
 input_length=500
)



tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(twenty_train.data)


X_train = tokenizer.texts_to_sequences(twenty_train.data)
X_test = tokenizer.texts_to_sequences(twenty_test.data)
X_train = pad_sequences(X_train, padding='post', maxlen=500)
X_test = pad_sequences(X_test, padding='post', maxlen=500)


model = Sequential()
model.add(embedding)
model.add(layers.Flatten())
model.add(layers.Dense(
 10,
 activation='relu',
 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
))
model.add(layers.Dense(len(categories), activation='softmax'))
model.compile(optimizer='adam',
 loss=SparseCategoricalCrossentropy(),
 metrics=['accuracy'])
model.summary()

model.fit(X_train, twenty_train.target, epochs=10)
predicted = model.predict(X_test).argmax(axis=1)
np.mean(predicted == twenty_test.target)



class FileIter(object):
  def __init(self, filepath: str):
   self.path = datapath(filepath)

  def __iter__(self):
    with utils.open(self.path, 'r', encoding='utf-8') as fin:
      for line in fin:
       yield list(tokenize(line))


model = FastText(size=4, window=3, min_count=1)
model.build_vocab(
 sentences=FileIter(
 'crime-and-punishment.txt'
))
model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)

model.wv['axe']

x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)

word_index = {i: w for i, w in enumerate(model.wv.vocab.keys())}


embedding_layer = Embedding(
 len(word_index) + 1,
 300,
 weights=[list(model.wv.vectors)],
 input_length=500,
 trainable=False
)


CONV_SAMPLES = {
 'greetings' : ['Hi', 'hello', 'How are you', 'hey there', 'hey'],
 'taxi' : ['book a cab', 'need a ride', 'find me a cab'],
 'weather' : ['what is the weather in tokyo', 'weather germany',
 'what is the weather like in kochi',
 'what is the weather like', 'is it hot outside'],
 'datetime' : ['what day is today', 'todays date', 'what time is it now',
 'time now', 'what is the time'],
 'music' : ['play the Beatles', 'shuffle songs', 'make a sound']
}

CLF = Classifier()
for key in CONV_SAMPLES:
 CLF.fit(CONV_SAMPLES[key], key)

print(CLF.predict('will it rain today')) # >>> 'weather'
print(CLF.predict('play playlist rock n\'roll')) # >>> 'music'
print(CLF.predict('what\'s the hour?')) # >>> 'datetime'

X_WEATHER = [
 'what is the weather in tokyo',
 'weather germany',
 'what is the weather like in kochi'
]
Y_WEATHER = [
 {'intent': 'weather', 'place': 'tokyo'},
 {'intent': 'weather', 'place': 'germany'},
 {'intent': 'weather', 'place': 'kochi'}
]

EX_WEATHER = EntityExtractor()
EX_WEATHER.fit(X_WEATHER, Y_WEATHER)

EX_WEATHER.predict('what is the weather in London')

{'intent': 'weather', 'place': 'London'}


mgr = OWM('YOUR-API-KEY').weather_manager()

def get_weather_forecast(place):
 observation = mgr.weather_at_place(place)
 return observation.get_weather().get_detailed_status()

print(get_weather_forecast('London'))

X_GREETING = ['Hii', 'helllo', 'Howdy', 'hey there', 'hey', 'Hi']
Y_GREETING = [{'greet': 'Hii'}, {'greet': 'helllo'}, {'greet': 'Howdy'},
 {'greet': 'hey'}, {'greet': 'hey'}, {'greet': 'Hi'}]
EX_GREETING = EntityExtractor()
EX_GREETING.fit(X_GREETING, Y_GREETING)

X_DATETIME = ['what day is today', 'date today', 'what time is it now', 'time now']
Y_DATETIME = [{'intent' : 'day', 'target': 'today'}, {'intent' : 'date', 'target': 'today'},
 {'intent' : 'time', 'target': 'now'}, {'intent' : 'time', 'target': 'now'}]

EX_DATETIME = EntityExtractor()
EX_DATETIME.fit(X_DATETIME, Y_DATETIME)

_EXTRACTORS = {
 'taxi': None,
 'weather': EX_WEATHER,
 'greetings': EX_GREETING,
 'datetime': EX_DATETIME,
 'music': None
}


_EXTRACTORS = {
 'taxi': None,
 'weather': EX_WEATHER,
 'greetings': EX_GREETING,
 'datetime': EX_DATETIME,
 'music': None
}

def question_and_answer(u_query: str):
 q_class = CLF.predict(u_query)
 print(q_class)
 if _EXTRACTORS[q_class] is None:
   return 'Sorry, you have to upgrade your software!'

 q_entities = _EXTRACTORS[q_class].predict(u_query)
 print(q_entities)
 if q_class == 'greetings':
   return q_entities.get('greet', 'hello')

 if q_class == 'weather':
   place = q_entities.get('place', 'London').replace('_', ' ')
   return 'The forecast for {} is {}'.format(
     place,
     get_weather_forecast(place)
 )

 if q_class == 'datetime':
   return 'Today\'s date is {}'.format(
     datetime.datetime.today().strftime('%B %d, %Y')
 )

 return 'I couldn\'t understand what you said. I am sorry.'

while True:
 query = input('\nHow can I help you?')
 print(question_and_answer(query))

[r'Is there (.*)', [
 "Do you think there is %1?",
 "It's likely that there is %1.",
 "Would you like there to be %1?"
]],

gReflections = {
 #...
 "i'd" : "you would",
}

p = Pattern('I want to eat [food: pizza, banana, yogurt, kebab]')
p('i\'d like to eat sushi')

{'food' : 'sushi'}

nvidia-smi

