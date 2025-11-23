import streamlit as st

st.set_page_config(
    page_title="AI Experiments Dashboard",
    layout="wide"
)

st.title("🌱 AI Experiments & CO₂ Impact Dashboard")

st.markdown("""
Welcome to the interactive dashboard.

Use the navigation menu on the left to access:

### 🏷️ Image Classification
Train MLP/CNN models on MNIST, visualize predictions, wrong predictions, emissions and losses.

### 🎨 Image Generation
Generate digits using CVAE (MLP/CNN), PCA or Probabilistic Pixel models.

### 💬 Text Generation
Run masked-word inference with BERT or prompt-based generation with GPT-2.

All experiments show **live CO₂ emissions** thanks to CodeCarbon.
""")
