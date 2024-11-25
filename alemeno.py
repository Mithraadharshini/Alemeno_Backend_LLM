# import streamlit as st
# from process_document import process_documents
# from chat_engine import query_documents

# # File paths for PDFs
# PDF_FILES = ["alphabet_10k.pdf", "tesla_10k.pdf", "uber_10k.pdf"]

# st.title("Content Engine with LangChain and Hugging Face")

# # Sidebar options
# if st.sidebar.button("Process PDFs"):
#     with st.spinner("Processing PDFs..."):
#         process_documents(PDF_FILES)
#         st.success("Documents processed and stored in ChromaDB!")

# st.header("Chat Interface")
# query = st.text_input("Ask a question:")
# if query:
#     with st.spinner("Fetching response..."):
#         response = query_documents(query)
#         st.success(response)


import streamlit as st
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import os

nltk.download('punkt')  # Ensure sentence tokenization works

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Pre-trained Sentence Transformer model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")  # Fine-tuned QA model

# Function to read PDF and extract text
def read_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Read and preprocess PDF files
# @st.cache(allow_output_mutation=True)
def load_documents_and_index(pdf_files):
    all_text = ""
    for pdf_file in pdf_files:
        all_text += read_pdf(pdf_file)

    # Split text into sentences using nltk for better accuracy
    document_sentences = sent_tokenize(all_text)

    # Generate embeddings for all sentences
    sentence_embeddings = embedding_model.encode(document_sentences, convert_to_tensor=False)

    # Initialize FAISS index
    dim = sentence_embeddings.shape[1]  # Embedding dimensions
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(sentence_embeddings))  # Add sentence embeddings

    return document_sentences, faiss_index

# Load PDF files (Replace with your paths)
pdf_files = ["goog-10-k-2023.pdf", "tsla-20231231-gen.pdf", "uber-10-k-2023.pdf"]
document_sentences, faiss_index = load_documents_and_index(pdf_files)

# Function to retrieve top_k relevant passages
def retrieve_passages(query, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=False)
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)

    # Combine retrieved sentences
    retrieved_sentences = " ".join([document_sentences[idx] for idx in indices[0]])
    return retrieved_sentences

# Function to generate a response
def generate_response(query):
    context = retrieve_passages(query, top_k=5)  # Retrieve relevant sentences
    response = qa_model(question=query, context=context)  # Use QA pipeline
    return response.get('answer', "No answer found.")

# Streamlit interface
st.title("Enhanced RAG Chatbot")
st.write("Ask questions about Google, Tesla, or Uber's Form 10-K filings!")

query = st.text_input("Enter your question:", "")
if query:
    response = generate_response(query)
    st.write(f"Answer: {response}")
