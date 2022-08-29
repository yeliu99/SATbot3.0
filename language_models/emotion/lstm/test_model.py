# pip install numpy==1.19.5
# pip install h5py==2.10.0
import string
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

str_punc = string.punctuation.replace(',', '').replace("'",'')

def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

tokenizer_file = open('/Users/yeliu/IC/Individual_Project/code/sat3.0/dl_models/emotion/lstm/tokenizer.pickle', 'rb')
le_file = open('/Users/yeliu/IC/Individual_Project/code/sat3.0/dl_models/emotion/lstm/labelEncoder.pickle', 'rb')
tokenizer = pickle.load(tokenizer_file)
le = pickle.load(le_file)
model = load_model('/Users/yeliu/IC/Individual_Project/code/sat3.0/dl_models/emotion/lstm/Emotion Recognition.h5')

sentence = 'fine'
sentence = clean(sentence)
sentence = tokenizer.texts_to_sequences([sentence])
sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
proba =  np.max(model.predict(sentence))
print(f"{result} : {proba}\n\n")