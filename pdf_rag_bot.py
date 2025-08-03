import pdfplumber
import numpy as np
import faiss
import requests
import streamlit as st 
from dotenv import load_dotenv
import os

load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")
EURI_CHAT_URL = os.getenv("EURI_CHAT_URL")
EURI_EMBED_URL = os.getenv("EURI_EMBED_URL")

conversation_memory = []

def extract_text_fromm_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embeddings(text):
    url = EURI_EMBED_URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    embedding = np.array(data['data'][0]['embedding'])
    
    return embedding

def build_vector_store(chunks):
    embeddings_list = []
    for chunk in chunks:
        emb = get_embeddings(chunk)  # each emb is shape (D,)
        embeddings_list.append(emb)

    embeddings = np.array(embeddings_list, dtype="float32")  # shape (N, D)
   # embeddings = get_embeddings(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrive_context(question, chunks, index, embeddings, top_k=1):
    q_embed = get_embeddings(question)
    D, I = index.search(np.array([q_embed]), top_k)
    return "\n\n".join([chunks[i] for i in I[0]])

def ask_llm_with_context(question, context, memory=True):
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering question from a document."}
    ]
    if memory:
        messages.extend(memory)
        
    messages.append({
        "role": 'user',
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    
    payload = {
        "messages": messages,
        "model": "gpt-4.1-nano",
        "temperature": 0.7
    }
    
    res = requests.post(EURI_CHAT_URL, headers=headers, json=payload)
    reply = res.json()['choices'][0]['message']['content']
    memory.append({"role": "user", "content":question})
    memory.append({"role": "system", "content":reply})
    
    return reply

# Streamlit UI
st.title("PDF Knowledege Extraction using RAG Bot")
uploaded_file = st.file_uploader("Upload a PDF", type = "pdf")
user_question = st.text_input("Ask a question about the document")

if uploaded_file:
    with open("temp.pdf", 'wb') as f:
        f.write(uploaded_file.read())
        full_text = extract_text_fromm_pdf("temp.pdf")
        chunks = split_text(full_text)
        index, embeddings = build_vector_store(chunks)
        
        st.success("PDF loaded and indexed")
        
        if user_question:
            context = retrive_context(user_question, chunks, index, embeddings)
            response = ask_llm_with_context(user_question, context, conversation_memory)
            print("Conversation Memory:", conversation_memory)
            st.markdown("### Answer: ")
            st.write(response)
            