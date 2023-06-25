import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd

st.title("Cognitive Level Identifier")
input_question = st.text_area("Enter Question")
question = [input_question]

if st.button('Predict'):

    df = pd.read_excel('Main Dataset.xlsx')
    df = df[['Label','Question']]
    df = df[pd.notnull(df['Question'])]
    df.rename(columns={'Question':'Question'}, inplace=True)

    from bs4 import BeautifulSoup
    def cleantext(text):

        text = re.sub(r'\|\|\|', r' ', text)
        text = re.sub(r'http\S+', r'<URL>', text)
        text = text.lower()
        text = text.replace('x', '')
        return text
    df['Question'] = df['Question'].apply(cleantext)

    df['Question'] = df['Question'].apply(cleantext)

    import nltk
    from nltk.corpus import stopwords
    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                #if len(word) < 0:
                if len(word) <= 0:
                    continue
                tokens.append(word.lower())
        return tokens



    max_fatures = 720437


    MAX_SEQUENCE_LENGTH = 60


    tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['Question'].values)
    X = tokenizer.texts_to_sequences(df['Question'].values)
    X = pad_sequences(X)

    loaded_model = load_model("LSTM.h5")



    #question = ['Social security  of Bangladesh is increasing. Criminals are being identified easily after the crime is commited. For this social life is being developed. How the mentionable life in the last  of the stem can be developed by ICT?']
    seq = tokenizer.texts_to_sequences(question)

    padded = pad_sequences(seq, 60, dtype='int32', value=0)

    pred = loaded_model.predict(padded)

    labels = ['C1', 'C2', 'C3', 'C4']
    result = labels[np.argmax(pred)]
    if result == 'C1':
        st.header("C1")
    elif result == 'C2':
        st.header("C2")
    elif result == 'C3':
        st.header("C3")
    else:
        st.header("C4")