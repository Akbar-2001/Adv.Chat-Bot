import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
import random
import time 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words

def clean_corpus(corpus):
  corpus = [ doc.lower() for doc in corpus]
  cleaned_corpus = []
  stop_words = stopwords.words('english')
  wordnet_lemmatizer = WordNetLemmatizer()
  for doc in corpus:
    tokens = word_tokenize(doc)
    cleaned_sentence = [] 
    for token in tokens: 
      if token not in stop_words and token.isalpha(): 
        cleaned_sentence.append(wordnet_lemmatizer.lemmatize(token)) 
    cleaned_corpus.append(' '.join(cleaned_sentence))
  return cleaned_corpus

import json
#java script object notation
with open('intents.json') as file:
    intents = json.load(file)
    
corpus = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        tags.append(intent['tag'])

        
cleaned_corpus = clean_corpus(corpus)
cleaned_corpus

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_corpus)

X.shape

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(np.array(tags).reshape(-1,1))


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
                    Dense(128, input_shape=(X.shape[1],), activation='relu'),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X.toarray(), y.toarray(), epochs=20, batch_size=1)

INTENT_NOT_FOUND_THRESHOLD = 0.40

def predict_intent_tag(message):
  message = clean_corpus([message])
  X_test = vectorizer.transform(message)
  y = model.predict(X_test.toarray())
  if y.max() < INTENT_NOT_FOUND_THRESHOLD:
    return 'noanswer'
  
  prediction = np.zeros_like(y[0])
  prediction[y.argmax()] = 1
  tag = encoder.inverse_transform([prediction])[0][0]
  return tag

print(predict_intent_tag('How you could help me?'))
print(predict_intent_tag('swiggy chat bot'))
print(predict_intent_tag('Where\'s my order'))


def get_intent(tag):
  for intent in intents['intents']:
    if intent['tag'] == tag:
      return intent
    
def perform_action(action_code, intent):
  
  if action_code == 'CHECK_ORDER_STATUS':
    print('\n Checking database \n')
    time.sleep(2)
    order_status = ['in kitchen', 'with delivery executive']
    delivery_time = []
    return {'intent-tag':intent['next-intent-tag'][0],
            'order_status': random.choice(order_status),
            'delivery_time': random.randint(10, 30)}
  
  elif action_code == 'ORDER_CANCEL_CONFIRMATION':
    ch = input('BOT: Do you want to continue (Y/n) ?')
    if ch == 'y' or ch == 'Y':
      choice = 0
    else:
      choice = 1
    return {'intent-tag':intent['next-intent-tag'][choice]}
  
  elif action_code == 'ADD_DELIVERY_INSTRUCTIONS':
    instructions = input('Your Instructions: ')
    return {'intent-tag':intent['next-intent-tag'][0]}
  
  while True:
  message = input('You: ')
  tag = predict_intent_tag(message)
  intent = get_intent(tag)
  response = random.choice(intent['responses'])
  print('Bot: ', response)

  if 'action' in intent.keys():
    action_code = intent['action']
    data = perform_action(action_code, intent)
    followup_intent = get_intent(data['intent-tag'])
    response = random.choice(followup_intent['responses'])

    if len(data.keys()) > 1:
      print('Bot: ', response.format(**data))
    else:
      print('Bot: ', response)

  if tag == 'goodbye':
    break
    
    
  
