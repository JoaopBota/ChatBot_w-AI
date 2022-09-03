##Train the neural network.....
import random #to implement a random number generator giving a a random response in the end
import json #for the JSon file
import pickle #for serialization,  the module implements binary protocols to the Python object structure, it is the process whereby a Python object hierarchy is converted into a byte stream
import numpy as np #Used to perform mathmatical operations in arrays

import nltk #natural language toolkit, used to process sentences and words(TEXT)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer #It is going to reduce the word  to its stem so we dont lose any performance looking for an exact word, basically to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form

from tensorflow.keras.models import Sequential #the sequential API allows to create linearly models layer-by-layer by passing a list of layer instances to the constructor trying to solve a problem.
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD #stochastic gradient descent is the method to optimize and smooth  the function with the properties below &&(see below)

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',','] #those are letters/signals that we wont take into account

#This code is related to what the AI defines as the class related to what the question was
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #what tokenize does it is it splits the sentences in individual words
        words.extend(word_list)
        documents.append((word_list,intent['tag'])) #this word list depends on this intent tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) # _set eliminates the duplicates and _sorted turns it back into a list

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#this code is related to the machine learning part in order to represnet this words in numerical values (code below)


training = []
output_empty = [0] * len(classes)
#when we run this all loop all will be on the training list to train the list
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words: #for each word we want to know if it s going to occur in the pattern
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training) #to shuffle the data
training = np.array(training)

train_x = list(training[:, 0]) #train everything in 0 dimension
train_y = list(training[:, 1]) #train erverything in Dimension 1

#adding all neural layers or neurons
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print("Done")

