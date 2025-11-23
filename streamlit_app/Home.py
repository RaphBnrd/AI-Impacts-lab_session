import streamlit as st

st.set_page_config(
    page_title="AI Experiments Dashboard",
    layout="wide"
)

st.title("🌱 AI Experiments & CO₂ Impact Dashboard")

st.markdown("""
Welcome to the interactive dashboard, to explore various AI experiments and their associated CO₂ emissions!
            
The environmental impact is assessed using the [CodeCarbon](https://codecarbon.io/) library.

### 🏷️ Image Classification
Train MLP/CNN models on the MNIST dataset and observe their performance and environmental impact.

→ *Explore [here](Image_Classification)*

### 🎨 Image Generation
Generate MNIST digits using CVAE (MLP/CNN), PCA or Probabilistic Pixel models.

→ *Explore [here](Image_Generation)*

### 💬 Text Generation
Run masked-word inference with BERT or prompt-based generation with GPT-2.

→ *Explore [here](Text_Generation)*
""")
