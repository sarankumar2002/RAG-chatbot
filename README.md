# RAG-chatbot
Chat with PDFs using RAG and Gemini AI

This project demonstrates to explore, processing PDF documents and store it into RAG vector search(FIASS) to provide user specific answers using Gemini Gen AI.

## Components
- **Streamlit:** Frontend framework to build Chatbot UI for user interactions
- **PyPDF2:** Library for reading PDF content
- **Langchain:** Library for NLP(Natural Language Processing)
- **Google Gen AI:** API is used for Gen AI to answer questions and maintain the context

## Features
- Users can upload multiple PDF files/documents
- Extract text form documents and create chunks for indexing
- Store chunks in vector in memory database using FIASS
- Users can ask questions related to PDF content using FIASS vector search and Gen AI

## Requirements
- Python 3.x
- **Libraries** are specified in ``requirement.txt``
  - python-dotenv 
  - google-generativeai 
  - python-dotenv 
  - langchain 
  - PyPDF2 
  - chromadb 
  - faiss-cpu 
  - langchain_google_genai 
  - streamlit 
  - langchain_community
- Google API Key configured in environment variables `GOOGLE_API_KEY`

## Installation steps
1. Clone the repository
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install the dependencies
   ```bash
    pip install -r requirements.txt
   ```
3. Setup environment variables
   Create `.env` file in the root directory and add your Google API key
   ```bash
      GOOGLE_API_KEY=<your_google_api_key>
   ```
## Run the application
```bash
    python -m streamlit run app.py
```
