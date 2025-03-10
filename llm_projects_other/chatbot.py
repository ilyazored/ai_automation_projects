import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import pyttsx3
import os

# CSS and HTML templates (unchanged from original code)
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://pic.onlinewebfonts.com/svg/img_24787.svg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    prompt_template = """
    You are an AI assistant expert at reading context information and answering questions based on the provided context and conversation history in english language.

    Context: {context}

    Conversation History: {chat_history}

    Human: {question}

    AI Assistant: Let's approach this step-by-step:

    1. First, I'll carefully review the context and our conversation history.
    2. Then, I'll formulate an answer based on the information provided.
    3. If there's not enough information in the context or our history, I'll mention that.
    4. I'll ensure my response is clear, concise, and directly addresses the question.

    Here's my response:
    """
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

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

def handle_userinput(user_question, use_tts=False):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    engine = pyttsx3.init() if use_tts else None

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            if use_tts and i == len(st.session_state.chat_history) - 1:
                engine.say(message.content)
                engine.runAndWait()

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question_text = st.text_input("Ask a question about your documents:")
    user_question_voice = get_user_input()
    
    use_tts = st.checkbox("Enable Text-to-Speech for responses")

    if user_question_text:
        handle_userinput(user_question_text, use_tts)
    elif user_question_voice:
        handle_userinput(user_question_voice, use_tts)

    with st.sidebar:
        st.subheader("Your documents")

if __name__ == '__main__':
    main()

