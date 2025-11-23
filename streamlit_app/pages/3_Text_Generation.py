import streamlit as st
import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from utils.processes import inference_bert, generate_gpt2

st.title("💬 Text Generation")

st.header("Model Selection")
mode = st.selectbox("Choose text model:", ["BERT (masked word)", "GPT-2 (prompt)"])

if mode == "BERT (masked word)":
    st.subheader("BERT: Masked Word Prediction")
    text = st.text_input("Enter a sentence with [MASK]:", "The capital of France is [MASK].")
    run = st.button("🚀 Run BERT")

    if run:
        st.info("Running BERT inference...")
        pred, emission = inference_bert(text)
        st.success("Done!")

        st.write("### Prediction")
        st.write(pred)

        st.subheader("Emissions")
        st.json(emission)

else:
    st.subheader("GPT-2: Text Generation")
    prompt = st.text_area("Enter a prompt:", "The future of AI is")
    run = st.button("🚀 Run GPT-2")

    if run:
        st.info("Generating text with GPT-2...")
        text_out, emission = generate_gpt2(prompt)
        st.success("Done!")

        st.write("### Output")
        st.write(text_out)

        st.subheader("Emissions")
        st.json(emission)
