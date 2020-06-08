import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words=['?','!']
data = open('intents.json').read()
intents= json.loads(data)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes=sorted(list(set(classes)))
print(len(documents),"documents")
print(len(classes),"classes",classes)
print(len(words),"unqiue lemmatized words",words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

tranning = []
output_empty = [0]*len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in words if w in words]
for w in words:
     bag.append(1) if w in pattern_words else bag.append(0)
output_row = list(output_empty)
output_row[classes.index(doc[1])]
tranning.append([bag,output_row])
random.shuffle(tranning)
tranning= np.array(tranning)
train_X = list(tranning[:,0])
train_Y = list(tranning[:,1])
print("traning data created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]),activation='softmax'))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


hist = model.fit(np.array(train_X),np.array(train_Y),epochs=200,batch_size=4,verbose=1)
model.save('chatbot_model.h5',hist)
print("model created")