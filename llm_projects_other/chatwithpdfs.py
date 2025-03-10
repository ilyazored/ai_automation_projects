import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("text_chunks is empty. Ensure the text is being split correctly.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    except IndexError as e:
        print("Error generating embeddings. The list of embeddings might be empty or incorrect.")
        print(f"text_chunks: {text_chunks}")
        raise e

    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
     I want you to act like an expert at reading context information and giving answer basis of context.
You are provided with the following
context : {context}
\n
Question : {question}

Do not make things up. Use the context that is provided to answer the question. If there is not enough information in the context then let the user know.

Answer:

    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Set allow_dangerous_deserialization=True to allow deserialization of the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with PDF using Gemini")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a question about the PDF files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click the Submit & Process button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                # st.write("Raw text extracted from PDF:")
                # st.write(raw_text[:500])  # Print the first 500 characters for debugging
                # st.write(f"Total length of extracted text: {len(raw_text)} characters")
                
                text_chunks = get_text_chunks(raw_text)
                # st.write("Text chunks created:")
                # for i, chunk in enumerate(text_chunks):
                #     st.write(f"Chunk {i}: {chunk[:500]}")  # Print the first 500 characters of each chunk for debugging
                # st.write(f"Total number of text chunks: {len(text_chunks)}")

                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
