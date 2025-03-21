import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
google_api = os.getenv("google")

# Configure Gemini model
genai.configure(api_key=google_api)
model = genai.GenerativeModel(model_name="gemini-1.5-flash-002")

# Function to extract text from PDFs
def pdf_texts(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

 
def vectors(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")

def search_query(query):
    vector_store = FAISS.load_local(
        "faiss_index",
        GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api),
        allow_dangerous_deserialization=True   
    )
    return vector_store.similarity_search(query)

def gemini_response(question, context):
    prompt = f"""Answer the question as detailed and clear as possible using the provided context.
    If the question is unrelated to the context, answer that you don't have specific information.
    
    Context: {context}
    Question: {question}
    """
    response = model.generate_content(prompt)
    return response.text

st.title("📄 AI-powered PDF Search with Gemini")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")
    
    extracted_text = pdf_texts(uploaded_files)
    text_chunks = split_text(extracted_text)
    vectors(text_chunks)
    
    st.success("Documents processed and stored successfully!")

query = st.text_input("Enter your search query:")

if query:
    st.write("🔍 Searching...")
    
    search_results = search_query(query)
    context = " ".join([doc.page_content for doc in search_results])
    
    response = gemini_response(query, context)
    
    st.subheader("🤖 AI Response:")
    st.write(response)
