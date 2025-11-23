import streamlit as st
import sys
import torch
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from transformers import BertTokenizer, BertForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils.processes import inference_bert, generate_gpt2
from utils.basics import print_emission_streamlit


# Device Selector

device_choice = st.selectbox(
    "Choose compute device",
    ["auto", "cpu", "cuda", "mps"]
)

def resolve_device(choice):
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"

    # AUTO
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = resolve_device(device_choice)
st.info(f"Using device: **{device}**")

# CONTENT
st.title("💬 Text Generation")

st.header("Model Selection")
mode = st.selectbox("Choose text model:", ["BERT (masked word)", "GPT-2 (prompt)"])

if mode == "BERT (masked word)":
    st.subheader("BERT: Masked Word Prediction")
    text = st.text_input("Enter a sentence with [MASK]:", "The capital of France is [MASK].")
    run = st.button("🚀 Run BERT")

    if run:
        st.info("Running BERT inference...")

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        num_params = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
        st.write(f"> BERT number of parameters: {num_params}")
        bert_model.eval()
        pred, emission = inference_bert(text, bert_model, bert_tokenizer, device)

        st.success("Done!")

        st.markdown(
            f"""
            <div style="font-size:22px; font-weight:600;">
                Prediction: <code>{pred}</code>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("**Full text:** "+text.replace("[MASK]", f"**{pred}**"))

        st.subheader("Emissions")
        print_emission_streamlit(emission)

else:
    st.subheader("GPT-2: Text Generation")
    prompt = st.text_area("Enter a prompt:", "The future of AI is")
    max_length = st.number_input("Max Length:", min_value=10, max_value=10000, value=200)
    run = st.button("🚀 Run GPT-2")

    if run:
        st.info("Generating text with GPT-2...")

        
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        num_params = sum(p.numel() for p in gpt2_model.parameters() if p.requires_grad)
        st.write(f"> GPT-2 number of parameters: {num_params}")
        gpt2_model.eval()
        text_out, nbr_tokens_generated, emission = generate_gpt2(
            prompt, gpt2_model, gpt2_tokenizer, device, max_length=max_length
        )
        
        st.success("Done!")

        st.write(f"**Output:** {text_out}")
        st.write(f"> Number of tokens generated: {nbr_tokens_generated}")

        st.subheader("Emissions")
        print_emission_streamlit(emission)
