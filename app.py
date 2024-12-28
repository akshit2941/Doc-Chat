import streamlit as st
from PyPDF2 import PdfReader
import os
import pickle
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME, cache_folder="./models")

# Load GPT model
TEXT_GEN_MODEL_NAME = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(TEXT_GEN_MODEL_NAME, cache_dir="./models")
text_gen_model = AutoModelForCausalLM.from_pretrained(TEXT_GEN_MODEL_NAME, cache_dir="./models")

# Function to extract embeddings
def get_embeddings(texts):
    """Generate embeddings for text chunks using Sentence Transformer."""
    return embedder.encode(texts, show_progress_bar=True)

# Function to generate answers
def generate_answer(context, query):
    """Generate an answer using the GPT model."""
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = text_gen_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Streamlit UI
def main():
    st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
    st.markdown(
        """
        <style>
        body {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .message-box {
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            max-width: 70%;
            font-size: 16px;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #4CAF50;
            color: #FFFFFF;
            margin-left: auto;
        }
        .doc-message {
            background-color: #2E2E2E;
            color: #FFFFFF;
            margin-right: auto;
        }
        #fixed-text-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 10px;
            background-color: #1E1E1E;
            border-top: 1px solid #444444;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📄 DocChat: Chat with Your PDF")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    with st.sidebar:
        st.header("📤 Upload Your PDF")
        pdf = st.file_uploader("Choose a PDF file", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
        
        # Split text into manageable chunks
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text=text)
        
        # Load or generate embeddings
        store_name = pdf.name[:-4]
        pkl_path = f"{store_name}.pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = get_embeddings(chunks)
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(np.array(embeddings))
            VectorStore = {"index": faiss_index, "chunks": chunks}
            with open(pkl_path, "wb") as f:
                pickle.dump(VectorStore, f)

        # Chat interface
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state["chat_history"]:
                if chat["type"] == "user":
                    st.markdown(
                        f"""
                        <div class="message-box user-message">
                            <b>You:</b> {chat['text']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="message-box doc-message">
                            <b>Doc:</b> <span id="doc-text-{len(st.session_state['chat_history'])}" class="typewriter">{chat['text']}</span>
                        </div>
                        <script>
                            var docText = document.getElementById('doc-text-{len(st.session_state['chat_history'])}');
                            var text = docText.innerText;
                            docText.innerText = '';
                            var i = 0;
                            var speed = 50;
                            function typeWriter() {{
                                if (i < text.length) {{
                                    docText.innerHTML += text.charAt(i);
                                    i++;
                                    setTimeout(typeWriter, speed);
                                }}
                            }}
                            typeWriter();
                        </script>
                        """,
                        unsafe_allow_html=True,
                    )

        user_input_area = st.empty()
        with user_input_area.container():
            user_query = st.text_input(
                "Type your question here...",
                key="user_query",
                label_visibility="collapsed",
                help="Ask a question about the document."
            )
            send_button = st.button("Send")

        # Handle query
        if send_button and user_query.strip():
            # User's query
            st.session_state["chat_history"].append({"type": "user", "text": user_query})

            # Compute embedding for the query
            query_embedding = embedder.encode([user_query])
            k = 5  # Number of top matches to retrieve
            distances, indices = VectorStore["index"].search(np.array(query_embedding), k)
            
            # Retrieve matching chunks
            retrieved_chunks = [VectorStore["chunks"][idx] for idx in indices[0]]
            context = "\n".join(retrieved_chunks)

            # Generate response
            response = generate_answer(context, user_query)
            st.session_state["chat_history"].append({"type": "doc", "text": response})

            st.rerun()

if __name__ == "__main__":
    main()
