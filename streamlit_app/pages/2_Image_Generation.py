import streamlit as st
import torch
import pandas as pd

import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from utils.processes import (
    train_cvae_MLP, train_cvae_CNN
)

from utils.basics import (
    plot_generate_digit,
    generate_digit_cvae, 
    train_PCA, generate_digit_pca,
    train_proba_pixel, generate_digit_proba_pixel, 

)

st.title("🎨 Image Generation")

device_choice = st.selectbox(
    "Select compute device:",
    ["auto", "cpu", "cuda", "mps"]
)

def resolve_device(choice):
    # if choice == "cpu":
    #     return "cpu"
    # if choice == "cuda":
    #     return "cuda" if torch.cuda.is_available() else "cpu"
    # if choice == "mps":
    #     return "mps" if torch.backends.mps.is_available() else "cpu"
    # auto
    if choice != "auto":
        return choice
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

st.header("Run Experiment")

model_type = st.selectbox("Choose generator:", [
    "CVAE (MLP)", "CVAE (CNN)", "PCA", "Probabilistic Pixel"
])

# Optional hyperparameters
if model_type.startswith("CVAE"):
    if model_type == "CVAE (MLP)":
        default_epochs = 10
        default_hidden = "400,100"
    else:
        default_epochs = 5
        default_hidden = "32,64,128"
    nbr_epochs = st.number_input("Epochs", min_value=1, max_value=30, value=default_epochs)
    latent_dim = st.number_input("Latent Dimension", min_value=1, value=20)
    hidden = st.text_input("Hidden Dimensions (comma-separated)", value=default_hidden)

run_gen = st.button("🚀 Generate")

if run_gen:
    status = st.empty()
    status.info("Launch experiment...")

    if model_type.startswith("CVAE"):
        hidden = list(map(int, hidden.split(",")))
        progress_bar = st.progress(0)
        progress_text = st.empty()

    if model_type == "CVAE (MLP)":
        status.info("Training CVAE (MLP)...")
        cvae, emission_train = train_cvae_MLP(
            hidden, latent_dim, train_loader, device, nbr_epochs=nbr_epochs,
            streamlit_progress=progress_bar, streamlit_text=progress_text,
            streamlit_status=status
        )
        emission_train['Number of Epochs'] = nbr_epochs
        status.info("Generating images...")
        imgs, emission_gen = generate_digit_cvae(
            cvae, n_samples=12, device=device, track_emissions=True, plot=False
        )

    elif model_type == "CVAE (CNN)":
        status.info("Training CVAE (CNN)...")
        cvae, emission_train = train_cvae_CNN(
            hidden, latent_dim, train_loader, device, nbr_epochs=nbr_epochs,
            streamlit_progress=progress_bar, streamlit_text=progress_text,
            streamlit_status=status
        )
        emission_train['Number of Epochs'] = nbr_epochs
        status.info("Generating images...")
        imgs, emission_gen = generate_digit_cvae(
            cvae, n_samples=12, device=device, track_emissions=True, plot=False
        )

    elif model_type == "PCA":
        status.info("Training PCA...")
        pcas, emission_train = train_PCA(
            X_train, y_train, n_components=50, track_emissions=True
        )
        status.info("Generating images...")
        imgs, emission_gen = generate_digit_pca(
            pcas, n_samples=12, track_emissions=True, plot=False
        )

    elif model_type == "Probabilistic Pixel":
        status.info("Training Probabilistic Pixel...")
        P, emission_train = train_proba_pixel(
            X_train, y_train, track_emissions=True
        )
        status.info("Generating images...")
        imgs, emission_gen = generate_digit_proba_pixel(
            P, n_samples=12, track_emissions=True, plot=False
        )

    st.success("Generation complete!")

    st.header("Emission Reports")

    def print_emission(emission_in):

        labels = {
            'run_name': "Experiment",
            'nbr_parameters': "Number of Parameters",
            'nbr_epochs': "Number of Epochs",
            'total_training_time_sec': "Total Training Time (s)",
            'final_train_loss': "Final Train Loss",
            'final_test_loss': "Final Test Loss",
            'final_train_accuracy': "Final Train Accuracy",
            'final_test_accuracy': "Final Test Accuracy",
            'total_emissions_kgCO2eq': "Total CO₂ Emissions (kg)",
            'total_energy_kWh': "Total Energy (kWh)",
            'total_cpu_energy_kWh': "Total CPU Energy (kWh)",
            'total_gpu_energy_kWh': "Total GPU Energy (kWh)",
            'total_ram_energy_kWh': "Total RAM Energy (kWh)",
            'cpu_power_W': "CPU Power (W)",
            'gpu_power_W': "GPU Power (W)",
            'ram_power_W': "RAM Power (W)",
            'cpu_model': "CPU Model",
            'gpu_model': "GPU Model",
            'total_water_L': "Total Water Usage (L)",
            'ram_total_size': "Total RAM Size (GB)"
        }

        emission = emission_in.copy()
        for key in emission:
            if isinstance(emission[key], float):
                if emission[key] < 0.01:
                    emission[key] = f"{emission[key]:.2e}"
                else:
                    emission[key] = f"{emission[key]:.3f}"
            if key not in labels:
                labels[key] = key
        emission = pd.DataFrame.from_dict(emission, orient='index', columns=['Value'])
        emission.index = emission.index.map(labels)
        st.table(emission)

    st.subheader("Training Emissions")
    print_emission(emission_train)
    st.subheader("Generation Emissions")
    print_emission(emission_gen)

    st.header("Generated Images")

    fig = plot_generate_digit(imgs)
    st.pyplot(fig)
