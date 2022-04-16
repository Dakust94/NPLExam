from eywa.nlu import Classifier
from eywa.nlu import EntityExtractor
from pyowm import OWM
import datetime

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

overcast clouds

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


##ELIZA

[r'Is there (.*)', [
    "Do you think there is %1?",
    "It's likely that there is %1.",
    "Would you like there to be %1?"
]],

gReflections = {
  #...
  "i'd" : "you would",
}

##EYWA

from eywa.nlu import Pattern

p = Pattern('I want to eat [food: pizza, banana, yogurt, kebab]')
p('i\'d like to eat sushi')

{'food' : 'sushi'}