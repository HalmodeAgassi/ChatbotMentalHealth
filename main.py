import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load the two pre-trained models and tokenizers
model_mental_health = TFBertForSequenceClassification.from_pretrained('./my_model_bert')
tokenizer_mental_health = BertTokenizer.from_pretrained('./my_model_bert')

model_gad7 = TFBertForSequenceClassification.from_pretrained('./gad7_bert_model')
tokenizer_gad7 = BertTokenizer.from_pretrained('./gad7_bert_model')

# Function to predict mental health conversation
def predict_mental_health(input_text):
    inputs = tokenizer_mental_health(input_text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    outputs = model_mental_health(**inputs)
    health_mental_pred = tf.argmax(outputs.logits, axis=-1).numpy()[0]
    return health_mental_pred

# Function to predict GAD-7 anxiety score
def predict_gad7(input_text):
    inputs = tokenizer_gad7(input_text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    outputs = model_gad7(**inputs)
    gad7_pred = tf.argmax(outputs.logits, axis=-1).numpy()[0]
    return gad7_pred

# Streamlit App Header
st.title("Mental Health Chatbot")
st.write("Welcome! Type your message below to chat with our mental health chatbot.")

# User input
user_input = st.text_input("You: ")

# Chatbot response based on user input
if user_input:
    # Predict mental health conversation
    health_mental_pred = predict_mental_health(user_input)
    
    if health_mental_pred == 0:
        st.write("Chatbot: Thank you for sharing. I'm here to listen and support you.")
    else:
        st.write("Chatbot: It seems like you're feeling anxious or stressed. Perhaps it's a good time to fill out the GAD-7.")
        st.write("Chatbot: Talking about your feelings or taking a short break might help you feel better.")

    # Predict GAD-7 anxiety level
    gad7_pred = predict_gad7(user_input)
    
    if gad7_pred == 0:
        st.write("Chatbot: There are no significant signs of anxiety. It's great that you're reaching out!")
    elif gad7_pred == 1:
        st.write("Chatbot: You seem to be slightly anxious. It might help to talk about what's on your mind, or consider taking a short break.")
        st.write("Chatbot: I'm here to listen if you want to share more.")
    elif gad7_pred == 2:
        st.write("Chatbot: You show moderate anxiety symptoms. It could help to engage in relaxation techniques, like deep breathing or taking a walk.")
        st.write("Chatbot: You might also find it helpful to talk to someone you trust or seek professional advice.")
    else:
        st.write("Chatbot: You seem to be experiencing more serious anxiety. I recommend seeking further support from a mental health professional or someone you trust.")
        st.write("Chatbot: You're not alone, and it's okay to ask for help. Your well-being matters.")

