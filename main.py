import re
import torch
import numpy as np
import json
import os
#import faiss
from config import *
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import LlamaForCausalLM, AutoTokenizer


print("connecting to mongo")
try:
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[MONGO_DB]
    collection = db[NEW_MONGO_COLLECTION]


    filtered_documents = collection.find()
    print(collection.count_documents())
    print("connected")
except Exception as e:
    print(e)
    

def preprocess_document(doc):
    #Cleans text from documents
    extracted_text = []
    for data_entry in doc.get("data", []):
        head = ' '.join(data_entry.get("head", []))
        text = ' '.join(data_entry.get("Text", []))
        combined_text = f"{head} {text}"
        cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()  # Remove excessive whitespace
        extracted_text.append(cleaned_text)
    return ' '.join(extracted_text)

def preprocess_documents(documents):
    #Preprocesses all the documents
    extracted_documents = []
    for document in documents:
        extracted_documents.append(preprocess_document(document))
    return extracted_documents

def load_processed_documents():
    #Load Processed documents from file if it exists
    if os.path.exists(PROCESSED_DOCS_FILE):
        with open(PROCESSED_DOCS_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_processed_documents(processed_docs):
    #Save the Processed documents to a file
    with open(PROCESSED_DOCS_FILE, 'w') as file:
        json.dump(processed_docs, file)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def create_vector_store(force_rebuild=False):
    #Create and Store Embeddings for documents in FAISS
    #Checks to see if a FAISS already exists
    if os.path.exists(FAISS_INDEX_DIR) and not force_rebuild:
        print("FAISS vector store already exists. Loading it instead of creating it.")
        return load_vector_store()

    print("Creating FAISS vector store...")
    processed_docs = load_processed_documents()

    if not processed_docs or force_rebuild:
        documents = collection.find()
        for doc in documents:
            if doc['url'] not in processed_docs:
                processed_docs[doc['url']] = preprocess_document(doc)
        save_processed_documents(processed_docs)

    texts = list(processed_docs.values())
    metadatas = [{"url": url} for url in processed_docs.keys()]
    vector_store = FAISS.from_texts(texts, embeddings, metadatas)
    vector_store.save_local(FAISS_INDEX_DIR)
    print("FAISS vector store created and saved locally.")
    return vector_store


def load_vector_store():
    #Loads the vecotr Store
    return FAISS.load_local(FAISS_INDEX_DIR, embeddings=embeddings, allow_dangerous_deserialization=True)

def query_vector_store(question, vector_store):
    #Queries the vector store for simialr documents
    results = vector_store.similarity_search(question, k=5)  # Retrieve top 5 most relevant documents
    context = "\n\n".join([result.page_content for result in results])
    return context

def load_llama_model():
    #Load the LLaMa model from Hugging Face
    print("Loading LLaMA model and tokenizer...")
    print(f"MODEL_NAME: {MODEL_NAME}, type: {type(MODEL_NAME)}")

    #Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("LLaMA model and tokenizer loaded successfully.")
    return tokenizer, model

def clean_response(response):
    #Clean up the respone generated, removing Context and the user input
    if "Advisor:" in response:
        response = response.split("Advisor:")[1]

    if "User:" in response:
        response = response.split("User:")[0] 

    return response.strip()

def generate_response(context, question, tokenizer, model):
    #Generates a response using LLama based on Queesiton and context
    prompt = (
        f"You are a helpful advisor. Use the following context to answer the user's question.\n\n"
        f"Context: {context}\n\nUser Question: {question}\n\nAdvisor:"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    #Token limit to increase responsiveness on my device
    output = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_new_tokens=100, 
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def chat_loop():
    #The loop for the chat advisor
    # Load models and vector store
    vector_store = create_vector_store()
    tokenizer, model = load_llama_model()
    
    print("Welcome to the Fullerton Advisor Chatbot!")
    print("Ask me any question related to CS at Fullerton. Type 'exit' to end.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        # User triggers reprocessing
        if user_input.lower() == 'rebuild index':
            print("Rebuilding vector store...")
            vector_store = create_vector_store(force_rebuild=True)
            continue
        
        #Get Context
        context = query_vector_store(user_input, vector_store)
        
        #Generate the advisor's response
        response = generate_response(context, user_input, tokenizer, model)
        #Clean the response
        response = clean_response(response)
        print("\nAdvisor:", response)


if __name__ == "__main__":
    chat_loop()