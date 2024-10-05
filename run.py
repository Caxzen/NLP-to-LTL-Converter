import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import streamlit as st
from tensorflow.keras.models import load_model
import pickle




data = pd.read_csv('dataset.csv')

model = load_model('lstm_model.h5')

# Load your tokenizer (ensure this is saved or available in your code)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Load the Label Encoder
with open('label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)


# Function to predict LTL formula from NLP statement
def convert_nlp_to_ltl(nlp_statements):
    sequences = tokenizer.texts_to_sequences(nlp_statements)
    padded_sequences = pad_sequences(sequences, maxlen=100)
    predicted_probabilities = model.predict(padded_sequences)
    
    # Get the index of the highest probability
    predicted_indices = np.argmax(predicted_probabilities, axis=1)
    
    # Map indices back to LTL formulas
    predicted_ltl = le.inverse_transform(predicted_indices)

    return predicted_ltl[0]




#interface

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             
             background: url("https://images.pexels.com/photos/12509859/pexels-photo-12509859.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");

             background-size: cover
             
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()

with st.sidebar:
        
    selected = option_menu('NLP to LTL Converter',['Converter',],
                            icons=['activity','heart','person'],
                            default_index=0)

st.header("NLP to LTL Converter")
nlp_input = st.text_input("Enter the NLP statement you want to convert:")

# Button to trigger the conversion
if st.button("Convert"):
    predicted_ltl = convert_nlp_to_ltl([nlp_input])
    st.code(predicted_ltl, language='python')

    #st.write(f"Predicted LTL Formula -> {predicted_ltl}")

# Example Area
st.subheader("Example NLP Statements")
st.write("Here are a few examples you can try:")

# Display the dataset as a table (optional)
if st.checkbox('Show Dataset'):
    st.dataframe(data)

st.write("This is a simple tool for converting natural language process (NLP) statements into Linear Temporal Logic (LTL) formulas.")



