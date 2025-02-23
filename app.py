import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import torch

class SimpleRAGChat:
    def __init__(self, json_path):
        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Initialize models
        self.init_models()
        
        # Create document store
        self.create_document_store()
        
    def init_models(self):
        # Load sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load text generation model
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_length=512
        )
    
    def create_document_store(self):
        # Create text documents from JSON data
        self.documents = []
        
        # Add product information
        for product in self.data['products']:
            doc = f"Product {product['id']}: {product['name']} - {product['description']} - Category: {product['category']} - Price: ${product['price']} - Stock: {product['stock']}"
            self.documents.append(doc)
        
        # Add order information
        for order in self.data['orders']:
            doc = f"Order {order['id']}: Customer {order['customer_name']} ordered on {order['date']} - Total: ${order['total']} - Status: {order['status']}"
            self.documents.append(doc)
        
        # Add customer information
        for customer in self.data['customers']:
            doc = f"Customer {customer['id']}: {customer['name']} - Email: {customer['email']} - Joined: {customer['join_date']} - Total Orders: {customer['total_orders']}"
            self.documents.append(doc)
        
        # Create FAISS index
        embeddings = self.embedding_model.encode(self.documents)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
    
    def retrieve_relevant_docs(self, query, k=2):
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        _, indices = self.index.search(query_embedding, k)
        
        # Return relevant documents
        return [self.documents[i] for i in indices[0]]
    
    def answer_query(self, query):
        # Get relevant context
        relevant_docs = self.retrieve_relevant_docs(query)
        context = "\n".join(relevant_docs)
        
        # Prepare prompt
        prompt = f"""
        Given this context about our store:
        {context}
        
        Answer this question: {query}
        
        Provide a clear and concise answer based on the available information.
        """
        
        # Generate answer
        response = self.generator(prompt, max_length=200)[0]['generated_text']
        return response, relevant_docs

def main():
    st.title("💬 Store Data Assistant")
    st.caption("Ask questions about products, orders, and customers")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize RAG chat
    if 'rag_chat' not in st.session_state:
        with st.spinner("Loading models... This might take a minute..."):
            st.session_state.rag_chat = SimpleRAGChat('./data.json')
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources if available
            if "sources" in message:
                with st.expander("View sources"):
                    for source in message["sources"]:
                        st.write(source)
    
    # Chat input
    if prompt := st.chat_input("Ask about the store data..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.rag_chat.answer_query(prompt)
                
                st.write(answer)
                
                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

if __name__ == "__main__":
    main()