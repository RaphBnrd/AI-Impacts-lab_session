import streamlit as st
import torch

import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from utils.processes import (
    train_cvae_MLP, train_cvae_CNN
)

from utils.basics import (
    generate_digit_cvae, train_PCA, generate_digit_pca,
    train_proba_pixel, generate_digit_proba_pixel
)

st.title("🎨 Image Generation")

device_choice = st.selectbox(
    "Select compute device:",
    ["auto", "cpu", "cuda", "mps"]
)

def resolve_device(choice):
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = resolve_device(device_choice)
st.info(f"Using device: **{device}**")

@st.cache_resource
def load_flat_mnist():
    from utils.processes import import_MNIST
    train_loader, test_loader, train_dataset, test_dataset = import_MNIST()
    X = train_dataset.data.numpy().reshape(-1, 28*28)
    y = train_dataset.targets.numpy()
    return train_loader, X, y

train_loader, X_train, y_train = load_flat_mnist()

st.header("Generator Options")

model_type = st.selectbox("Choose generator:", [
    "CVAE (MLP)", "CVAE (CNN)", "PCA", "Probabilistic Pixel"
])

# Optional hyperparameters
if model_type.startswith("CVAE"):
    nbr_epochs = st.number_input("Epochs", min_value=1, max_value=30, value=3)
else:
    nbr_epochs = 1  # PCA and ProbPixel don't use epochs

run_gen = st.button("🚀 Generate")

if run_gen:
    st.info("Running generation...")

    if model_type == "CVAE (MLP)":
        cvae, _ = train_cvae_MLP([400, 100], 20, train_loader, device, nbr_epochs=nbr_epochs)
        imgs, emission = generate_digit_cvae(cvae, n_samples=12, device=device, track_emissions=True)

    elif model_type == "CVAE (CNN)":
        cvae, _ = train_cvae_CNN([32, 64, 128], 20, train_loader, device, nbr_epochs=nbr_epochs)
        imgs, emission = generate_digit_cvae(cvae, n_samples=12, device=device, track_emissions=True)

    elif model_type == "PCA":
        pcas, _ = train_PCA(X_train, y_train, n_components=50, track_emissions=True)
        imgs, emission = generate_digit_pca(pcas, n_samples=12, track_emissions=True)

    elif model_type == "Probabilistic Pixel":
        P, _ = train_proba_pixel(X_train, y_train, track_emissions=True)
        imgs, emission = generate_digit_proba_pixel(P, n_samples=12, track_emissions=True)

    st.success("Generation complete!")

    st.subheader("Generated Images")
    cols = st.columns(6)
    for i, img in enumerate(imgs):
        cols[i % 6].image(img, use_column_width=True)

    st.subheader("Emission Report")
    st.json(emission)
