import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.append("../")
sys.path.append("./")

from utils.processes import (
    import_MNIST,
    expe_simple_MLP, expe_simple_CNN,
    train_cvae_MLP, train_cvae_CNN,
    inference_bert, generate_gpt2
)

from utils.basics import (
    plot_examples_classification, plot_wrong_classification,
    plot_loss_acc_over_epochs, plot_co2_energy_over_epochs,
    generate_digit_cvae, train_PCA, generate_digit_pca,
    train_proba_pixel, generate_digit_proba_pixel
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Experiments Dashboard",
    layout="wide"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Image Classification",
    "Image Generation",
    "Text Generation"
])

# --------------------------------------------------
# LOAD MNIST ONCE
# --------------------------------------------------
@st.cache_resource
def load_mnist():
    return import_MNIST(batch_size_train=64)

train_loader, test_loader, train_dataset, test_dataset = load_mnist()

# --------------------------------------------------
# PAGE 1: IMAGE CLASSIFICATION
# --------------------------------------------------
if page == "Image Classification":
    st.title("🖼️ Image Classification Experiments")

    st.sidebar.subheader("Run Experiments")

    exp_type = st.sidebar.selectbox("Model type", ["MLP", "CNN"])

    run_exp = st.sidebar.button("Run Experiment")

    if run_exp:
        st.info("Running experiment...")

        if exp_type == "MLP":
            hidden_dims_tested = [
                [64, 32],
                [128, 64],
                [256, 128, 64]
            ]
            res_epochs, summaries, all_models, labels = expe_simple_MLP(
                hidden_dims_tested, train_loader, test_loader, device, nbr_epochs=5
            )
        else:
            hidden_channels_tested = [
                [8], [32], [32, 32]
            ]
            res_epochs, summaries, all_models, labels = expe_simple_CNN(
                hidden_channels_tested, train_loader, test_loader, device, nbr_epochs=5
            )

        st.success("Experiment completed!")

        model_idx = st.selectbox("Select a model to visualize", list(range(len(all_models))))

        # Classification plots
        st.subheader("Predictions")
        fig1 = plt.figure()
        plot_examples_classification(all_models[model_idx], test_dataset, device, n=5)
        st.pyplot(fig1)

        st.subheader("Wrong Predictions")
        fig2 = plt.figure()
        plot_wrong_classification(all_models[model_idx], test_dataset, device, n=5)
        st.pyplot(fig2)

        st.subheader("Loss / Accuracy Over Epochs")
        fig3 = plot_loss_acc_over_epochs([res_epochs[model_idx]], [labels[model_idx]])
        st.pyplot(fig3)

        st.subheader("CO₂ / Energy Over Epochs")
        fig4 = plot_co2_energy_over_epochs([res_epochs[model_idx]], [labels[model_idx]])
        st.pyplot(fig4)

# --------------------------------------------------
# PAGE 2: IMAGE GENERATION
# --------------------------------------------------
elif page == "Image Generation":
    st.title("🎨 Image Generation Experiments")

    st.sidebar.subheader("Select Model")

    gen_type = st.sidebar.selectbox("Generation method", ["CVAE (MLP)", "CVAE (CNN)", "PCA", "Probabilistic Pixel"])

    run_gen = st.sidebar.button("Generate Samples")

    if run_gen:
        st.info("Running generation...")

        if gen_type == "CVAE (MLP)":
            cvae, _ = train_cvae_MLP([400, 100], 20, train_loader, device, nbr_epochs=3)
            imgs, emission = generate_digit_cvae(cvae, n_samples=12, device=device, track_emissions=True)

        elif gen_type == "CVAE (CNN)":
            cvae, _ = train_cvae_CNN([32, 64, 128], 20, train_loader, device, nbr_epochs=3)
            imgs, emission = generate_digit_cvae(cvae, n_samples=12, device=device, track_emissions=True)

        elif gen_type == "PCA":
            all_pcas, _ = train_PCA(
                train_dataset.data.numpy().reshape(-1, 28*28),
                train_dataset.targets.numpy(),
                n_components=50,
                track_emissions=True
            )
            imgs, emission = generate_digit_pca(all_pcas, n_samples=12, track_emissions=True)

        elif gen_type == "Probabilistic Pixel":
            all_p, _ = train_proba_pixel(
                train_dataset.data.numpy().reshape(-1, 28*28),
                train_dataset.targets.numpy(),
                track_emissions=True
            )
            imgs, emission = generate_digit_proba_pixel(all_p, n_samples=12, track_emissions=True)

        st.success("Generation complete!")

        # Show images
        cols = st.columns(6)
        for i, img in enumerate(imgs):
            cols[i % 6].image(img, width=100)

        st.subheader("Emission Info")
        st.json(emission)

# --------------------------------------------------
# PAGE 3: TEXT GENERATION
# --------------------------------------------------
elif page == "Text Generation":
    st.title("💬 Text Generation Experiments")

    st.sidebar.subheader("Choose Model")
    txt_type = st.sidebar.selectbox("Text model", ["BERT", "GPT-2"])

    if txt_type == "BERT":
        text = st.text_input("Enter a masked sentence:", "The capital of France is [MASK].")
        run = st.button("Run inference")

        if run:
            predicted, emission = inference_bert(text)
            st.write("### Prediction:", predicted)
            st.subheader("Emission info")
            st.json(emission)

    if txt_type == "GPT-2":
        prompt = st.text_area("Enter a prompt:", "Once upon a time...")
        run = st.button("Generate text")

        if run:
            text, emission = generate_gpt2(prompt)
            st.write("### Output:", text)
            st.subheader("Emission info")
            st.json(emission)
