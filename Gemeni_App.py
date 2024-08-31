import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# To get the pdf text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
    # Assuming 'split' has been replaced by 'split_text' or similar method
    chunks = text_splitter.split_text(text)  # Updated method name
    return chunks

def get_vector_store(text_chunks, api_key):
    genai.configure(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    You are a knowledgeable assistant who provides accurate and detailed answers based on the provided context. 
    If the answer is not explicitly available in the context, say, "The answer is not available in the context." 
    Do not generate misleading or incorrect information.

    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    
    genai.configure(api_key=api_key)
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"], output_variable="answer")
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def clean_response(response_text):
    # Example of basic post-processing
    cleaned_text = response_text.strip()
    if cleaned_text.lower().startswith("the answer is not available in the context"):
        return cleaned_text
    # Additional processing logic can be added here
    return cleaned_text


def user_input(user_question, api_key):
    genai.configure(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Load the FAISS index with dangerous deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    cleaned_response = clean_response(response["output_text"])
    st.write("Reply: ", cleaned_response)


def main():
    st.set_page_config(page_title="CheemsHUB")
    st.header("Chat with PDF using Gemini")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        api_key = st.text_input("Enter your Gemini API key", type="password")
        
        if st.button("Submit & Process"):
            if not api_key:
                st.error("Please enter a valid Gemini API key.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Done")
    
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and api_key:
        user_input(user_question, api_key)

if __name__ == "__main__":
    main()
