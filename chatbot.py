import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import tkinter # Library to create the GUI
from tkinter import *

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb')) #load the word class
classes = pickle.load(open('classes.pkl', 'rb')) #load the classes class
model = load_model('chatbotmodel.h5') #load the model class

# now we need to set up different  functions to use the right wy  because the model it is already trained and ready to use but what we have now is numerical data and we need to end up with words but it is
def clean_up_sentence(sentence): #function to cleaning up the sentence
    sentence_words = nltk.word_tokenize(sentence) #tokenize the sentence
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] #lemmatize the sentance
    return sentence_words


def bag_of_words(sentence): #a function to get the bag of words, it will convert a sentence into a list of 0's and 1's that indicate the word is there or not
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model): # a function to predict the class based on the sentence essentially, it will predict the class based on the sentence
    bow = bag_of_words(sentence) #thats what we need to feed into the neural network in order to get a prediction
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25 #to specify an error threshold that allows for certain uncertainty   to dont be shown in the results if the uncertainty is to high (in this case is lower than '0.25')
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True) #to sort the results// and the key for sorting the results will be an anonimous function, an lambda expression
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# So, basically we create a bag of words, we predict the result based on those bag of words  then we have a certain threshold to dont have too much uncertainty and we do is to enumerate
#  all the results in order to get the index to the class and also the probability. we sort the probability from the highest probable to the least probable and then we
#  want to have a return list  of intents and probabilities, full of classes and probabilities

def get_response(intent_list, intents_json):
    tag = intent_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    intent_list = predict_class(msg, model)
    resp = get_response(intent_list, intents)
    return resp


def send(): #GUI send request
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(font=("Acumin Variable Concept", 10, 'bold'))

        resp = chatbot_response(msg)
        ChatLog.insert(END, "ChatBot: " + resp + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

#All the properties below are related to the GUI ,background colours, type of text , type button,etc...
base = Tk()

base.iconbitmap(r'icon.ico')

base.title("ChatBot")

base.geometry("400x500")

base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, wrap="word", bg="white", height="10", width="50", font=("Acumin Variable Concept", 10, 'bold'), )
ChatLog.insert(END, "ChatBot: Hello, I'm Clirec also known as Client restaurant ChatBot" + '\n\n')
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Acumin Variable Concept", 12, 'bold'), text="Send Message", width="12", height=5, bd=0, bg="#328da8", activebackground="#707070", fg='#ffffff',command=send)

EntryBox = Text(base, bd=3, bg="white", width="60", height="5", font=("Acumin Variable Concept", 10, 'bold',))

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=30, y=6, height=430, width=320)
EntryBox.place(x=145, y=460, height=30, width=220)
SendButton.place(x=6, y=460, height=30)


base.mainloop()
