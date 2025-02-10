# Importing Libraries

import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# loading imdb data and word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# loading the pre-trained model trained with ReLU activation function
model=load_model('simple_rnn_imdb.h5')

# Function for decoding reviews
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function for preprocessing the user input
def preprocess_text(user_input):
  words=user_input.lower().split()
  encoded_review=[word_index.get(word,2)+3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review


# PREDICTION function
def predict_sentiment(review):
  preprocessed_input=preprocess_text(review)
  prediction=model.predict(preprocessed_input)
  sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
  return sentiment,prediction[0][0]

# streamlit app
st.title("IMDB Movie Review Sentiment Analyis")
st.write('Enter a movie review to classify whether it is positive or negative')

# user input
user_input=st.text_area('Movie Review')

if(st.button('Classify')):
  # make prediction
  prediction,sentiment=predict_sentiment(user_input)
  # Display the result
  st.write(f'Sample Review: {user_input}')
  st.write(f'Sentiment: {sentiment}')
  st.write(f'Prediction Score:{prediction}')
else:
  st.write('Please Enter a movie review.')
