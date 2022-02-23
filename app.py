import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer  # loading pretrained DistilBERT tokenizer
from transformers import TFDistilBertForSequenceClassification # loading pretrained DIstilBERT base uncased model

save_directory = "model_weights"
#save_directory = "https://drive.google.com/drive/folders/1-zhuAuyYUz9HscjRwPauu-J-dbY-zjta?usp=sharing"
loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

st.title("Balsam review to rating prediction model")

def predict_rating(input_review):
      predict_input = loaded_tokenizer.encode(input_review,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="tf")

      output = loaded_model(predict_input)[0]

      prediction_value = tf.argmax(output, axis=1).numpy()[0]
      predicted_rating = prediction_value+1

      return predicted_rating

input_review = st.text_input('Input your review here:') 
if input_review!="":
    predicted_rating = predict_rating(input_review)
    st.write("Predicted rating:  ",predicted_rating)
# elif(input_review==""):
#    st.write("*Input review should not be empty")
