import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

load_dotenv()

# Function to get text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks, api_key):
    if not text_chunks:
        st.error("No text chunks found. Please check your PDF content.")
        return

    # Initialize OpenAI embeddings with the GPT API key
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    try:
        # Generate embeddings for text chunks
        embedding_vectors = embeddings.embed_documents(text_chunks)
        
        if not embedding_vectors or len(embedding_vectors) == 0:
            st.error("Failed to generate embeddings. Check your API key or the content.")
            return
        
        # Check dimensions of the embeddings
        embedding_dim = len(embedding_vectors[0])
        st.write(f"Embedding dimensions: {embedding_dim}")
        
        # Create the FAISS vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")


# Function to create conversational chain with GPT
def get_conversational_chain(api_key):
    prompt_template = """
    You are a knowledgeable assistant who provides accurate and detailed answers based on the provided context. 
    If the answer is not explicitly available in the context, say, "The answer is not available in the context." 
    Do not generate misleading or incorrect information.

    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    
    model = ChatOpenAI(api_key=api_key, model="gpt-4", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"], output_variable="answer")
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to clean response text
def clean_response(response_text):
    cleaned_text = response_text.strip()
    if cleaned_text.lower().startswith("the answer is not available in the context"):
        return cleaned_text
    return cleaned_text

# Safe request function with retry logic for rate limits
def safe_request(api_func, *args, **kwargs):
    while True:
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            if 'Rate limit exceeded' in str(e):
                print(f"Rate limit exceeded: {e}")
                time.sleep(60)  # Wait for 1 minute before retrying
            else:
                raise

# Function to handle user input and generate responses
def user_input(user_question, api_key):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Limit the number of documents retrieved
    docs = new_db.similarity_search(user_question, k=5)

    chain = get_conversational_chain(api_key)

    response = safe_request(
        chain,
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    cleaned_response = clean_response(response["output_text"])
    st.write("Reply: ", cleaned_response)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="CheemsHUB")
    st.header("Chat with PDF using GPT-4")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        api_key = st.text_input("Enter your GPT API key", type="password")
        
        if st.button("Submit & Process"):
            if not api_key:
                st.error("Please enter a valid GPT API key.")
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
