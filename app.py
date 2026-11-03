import streamlit as st
from transformers import AutoTokenizer

st.set_page_config(page_title="LLM Pipeline Visualizer", page_icon="🧠", layout="wide")

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

tokenizer = load_tokenizer()

st.title("🧠 LLM Pipeline Visualizer")
st.caption("See what happens between hitting send and an LLM's answer.")

text = st.text_input("Type a sentence:", "What is the capital of France?")

if text:
    # Step 1: Tokenizer
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([i]) for i in token_ids]

    st.subheader("1. Tokenizer: text → tokens")
    st.write(" ".join(f"`{t}`" for t in tokens))

    # Step 2: Token IDs
    st.subheader("2. Token IDs: tokens → numbers")
    st.write("The model never sees words, only these numbers:")
    st.dataframe({"Token": tokens, "Token ID": token_ids}, use_container_width=True)