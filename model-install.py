from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="./models")

model_name = "EleutherAI/gpt-neo-125M"  # Or "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
text_gen_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models")
