import streamlit as st
from PyPDF2 import PdfReader
import os
import pickle
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model locally
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME, cache_folder="./models")

# Load local GPT model for text generation
TEXT_GEN_MODEL_NAME = "EleutherAI/gpt-neo-125M"  # You can also use "gpt2"
tokenizer = AutoTokenizer.from_pretrained(TEXT_GEN_MODEL_NAME, cache_dir="./models")
text_gen_model = AutoModelForCausalLM.from_pretrained(TEXT_GEN_MODEL_NAME, cache_dir="./models")

# Function to extract embeddings
def get_embeddings(texts):
    """Generates embeddings for text chunks using Sentence Transformer."""
    return embedder.encode(texts, show_progress_bar=True)

# Function to generate answers
def generate_answer(context, query):
    """Generate an answer using a local GPT model."""
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Handle the `max_length` issue with `max_new_tokens`
    outputs = text_gen_model.generate(
        **inputs,
        max_new_tokens=150,  # Generate up to 150 new tokens
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main function
def main():
    st.title("Chat with Your PDF")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf", key="pdf_uploader")
    
    if pdf is not None:
        # Parse PDF and extract text
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split text into chunks
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        # Check if embeddings already exist on disk
        store_name = pdf.name[:-4]
        pkl_path = f"{store_name}.pkl"
        
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                VectorStore = pickle.load(f)
            st.write("Loaded embeddings from disk.")
        else:
            # Generate embeddings using Sentence Transformer
            embeddings = get_embeddings(chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance
            faiss_index.add(np.array(embeddings))  # Add embeddings to the index
            
            # Save FAISS index and chunks as a dictionary
            VectorStore = {"index": faiss_index, "chunks": chunks}
            
            # Save vector store to disk
            with open(pkl_path, "wb") as f:
                pickle.dump(VectorStore, f)
            st.write("Generated and saved new embeddings.")

        # Display chat history
        st.subheader("Chat History")
        for chat in st.session_state["chat_history"]:
            st.markdown(f"**You:** {chat['query']}")
            st.markdown(f"**PDF:** {chat['response']}")
            st.markdown("---")

        # Input for the new question
        st.subheader("Ask a question")
        user_query = st.text_area(
            "Type your question here and press Enter to submit:",
            key="user_query",
            height=70,
        )
        
        # Process user query
        if st.button("Send"):
            if user_query.strip():
                # Compute embedding for the query
                query_embedding = embedder.encode([user_query])
                
                # Search FAISS index for the top-k similar chunks
                k = 3  # Number of top matches to retrieve
                distances, indices = VectorStore["index"].search(np.array(query_embedding), k)
                
                # Retrieve matching chunks
                retrieved_chunks = [VectorStore["chunks"][idx] for idx in indices[0]]
                context = "\n".join(retrieved_chunks)
                
                # Generate an answer using the local model
                response = generate_answer(context, user_query)
                
                # Update chat history
                st.session_state["chat_history"].append({"query": user_query, "response": response})
                st.experimental_rerun()

if __name__ == "__main__":
    main()
