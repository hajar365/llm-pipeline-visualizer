import json
import math
import random
import re
import zlib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="LLM Pipeline Visualizer", page_icon="🧠", layout="wide")

FEW_SHOT = (
    "Question: Is Tokyo the capital of Japan?\nAnswer: Yes, Tokyo is the capital of Japan.\n\n"
    "Question: What is the capital of Italy?\nAnswer: The capital of Italy is Rome.\n\n"
    "Question: Is the Earth flat?\nAnswer: No, the Earth is round.\n\n"
    "Question: Is Mount Everest the tallest mountain?\nAnswer: Yes, Mount Everest is the tallest mountain on Earth.\n\n"
    "Question: Why do birds sing?\nAnswer: Birds sing to attract mates and mark their territory.\n\n"
    "Question: Can penguins fly?\nAnswer: No, penguins cannot fly.\n\n"
)


def continue_answers(tok, model, full_ids, first_ids, temp, seed_text, max_new=20):
    """Generate the rest of the answer for ALL candidate first tokens in ONE batched call."""
    import torch

    torch.manual_seed(zlib.crc32(seed_text.encode()))
    batch = torch.tensor([full_ids + [i] for i in first_ids])  # same length -> no padding needed
    kw = dict(do_sample=True, temperature=temp, top_k=50) if temp > 0 else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(
            batch,
            attention_mask=torch.ones_like(batch),
            max_new_tokens=max_new,
            eos_token_id=[198, 628, 50256],  # newline, double newline, <|endoftext|>
            pad_token_id=50256,
            **kw,
        )
    return [
        tok.decode(o[batch.shape[1]:], skip_special_tokens=True).split("\n")[0].rstrip() for o in out
    ]


@st.cache_resource(show_spinner="Loading GPT-2 (first run downloads ~500 MB)…")
def load_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return tok, model


@st.cache_data(show_spinner="GPT-2 is thinking…")
def build_demo(text: str, temp: float, n_heads: int = 4, dim: int = 8, top_k: int = 6) -> dict:
    """Tokens, IDs and next-token probabilities come from real GPT-2.
    Embedding values and attention maps are still illustrative."""
    import torch

    tok, model = load_gpt2()
    prompt = f"Question: {text}\nAnswer:"  # what the user sees in steps 1-2
    ids = tok.encode(prompt)
    tokens = [tok.decode([i]).replace("\n", "⏎") for i in ids]
    # GPT-2 only continues text: a short hidden Q&A example makes it answer instead of ramble
    full_ids = tok.encode(FEW_SHOT) + ids

    with torch.no_grad():
        logits = model(torch.tensor([full_ids])).logits[0, -1]  # scores for the NEXT token
    if temp == 0:  # greedy
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
    else:
        probs = torch.softmax(logits / temp, dim=-1)
    top_p, top_i = torch.topk(probs, top_k)

    outs = continue_answers(tok, model, full_ids, top_i.tolist(), temp, text)

    rng = random.Random(zlib.crc32(text.encode()))
    emb = [[round(rng.uniform(-1, 1), 2) for _ in range(dim)] for _ in tokens]
    attn = []
    for _ in range(n_heads):
        head = []
        for i in range(len(tokens)):
            row = [math.exp(rng.uniform(0, 3)) for _ in range(i + 1)]
            s = sum(row)
            head.append([round(v / s, 3) for v in row])
        attn.append(head)

    return {
        "tokens": tokens,
        "ids": ids,
        "emb": emb,
        "attn": attn,
        "probs": [
            {"t": tok.decode([i]).replace("\n", "⏎"), "p": round(p, 4), "out": o}
            for p, i, o in zip(top_p.tolist(), top_i.tolist(), outs)
        ],
    }


st.title("🧠 LLM Pipeline Visualizer")
st.caption("From “send” to answer: every step of an LLM, animated. (Tokens, IDs and probabilities are real GPT-2; embeddings and attention are illustrative.)")

examples = [
    "is the capital of france paris?",
    "How are you doing today?",
    "Why is the sky blue?",
    "What is the meaning of life?",
    "Can you tell me a joke?",
    "How do I learn to code?",
]
c1, c2 = st.columns([2, 1])
choice = c1.selectbox("Try an example", examples)
temperature = c2.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
text = st.text_input("…or type your own", choice, max_chars=80)

html = Path(__file__).with_name("pipeline.html").read_text(encoding="utf-8")
data = json.dumps(build_demo(text, temperature)).replace("</", "<\\/")
html = html.replace("<!--DATA-->", f"<script>window.DEMO={data}</script>")
components.html(html, height=2250, scrolling=True)