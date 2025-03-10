from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
import speech_recognition as sr
import json
import pyttsx3


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model=genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response(input,image,prompt):
    response=model.generate_content([input,image[0],prompt])
    return response.text

def get_user_input():
    st.sidebar.write("Click the button below and start speaking...")
    if st.sidebar.button("Speak"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            st.write("Processing...")

        try:
            user_input = r.recognize_google(audio)
            return user_input
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand what you said.")
            return None
        except sr.RequestError as e:
            st.write("Could not request results from Google Speech Recognition service; {0}".format(e))
            return None

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
            "mime_type": uploaded_file.type,
            "data":bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


st.set_page_config("Gemini Application")

uploaded_file = st.file_uploader("Choose a image..",type=["jpg","jpeg","png"])
image=""
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="uploaded Image.",use_column_width=True)

input_prompt="""
You are an expert in understanding images.We will upload a image as
input and you will have to answer any questions based on the uploaded image carefully.
"""

input = st.text_input("Ask a question about this image")

submit=st.button("Tell me about the invoice")

if submit:
    image_data=input_image_setup(uploaded_file)
    response=get_gemini_response(input_prompt,image_data,input)
    st.write(response)
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()
