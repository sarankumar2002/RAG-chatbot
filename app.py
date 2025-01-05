import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from functions import *

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google api key in genai
genai.configure(api_key=api_key)
# Google model variables
GOOGLE_GEN_AI_MODEL = "gemini-pro"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"


# get prompt chain from Gemini AI
def get_response_from_genai():
    prompt_template = """

    Context: \n{context}?\n
    Question: \n{question}\n

    Answer:
    """
    # chat with Google Gen AI models
    model = ChatGoogleGenerativeAI(model=GOOGLE_GEN_AI_MODEL, temperature=0.2)
    # Create Prompt Template to establish conversational chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain
    prompt_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return prompt_chain


def process_user_input(user_question):
    # Google Gen Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL, google_api_key=api_key, transport="rest")
    # Create in memory database to store vector search
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # perform the similarity search
    docs = new_db.similarity_search(user_question)
    # Get the response from GenAI
    chain = get_response_from_genai()
    # Chain response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: \n ", response["output_text"])


# vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL, google_api_key=api_key, transport="rest")
    # In memory database
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# main class
def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Gemini...")
    # Ask questions from PDF files
    user_input = st.text_input("Ask a Question from the PDF Files")

    if user_input:
        process_user_input(user_input)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Multiple PDF files",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Uploaded Successfully!")


if __name__ == "__main__":
    main()
