import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import torch
import os
import json

import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from utils.processes import import_MNIST, plot_example_images, expe_simple_MLP, expe_simple_CNN
from utils.basics import (
    plot_examples_classification, plot_wrong_classification,
    sort_keys_expe
)
from utils.models import SimpleMLP, SimpleCNN

# --------------------------------------------------------------------
# PAGE TITLE
# --------------------------------------------------------------------
st.title("🏷️ Image Classification — Experiments Dashboard")

# --------------------------------------------------------------------
# LOAD MNIST
# --------------------------------------------------------------------
@st.cache_resource
def load_mnist():
    return import_MNIST(batch_size_train=64)

train_loader, test_loader, train_dataset, test_dataset = load_mnist()


# ===============================================================
# ===========  SECTION 0 — EXAMPLE DATASET IMAGES  ==============
# ===============================================================

st.header("🖼️ Example MNIST Images")

fig = plot_example_images(test_dataset, n=5)
st.pyplot(fig)


# ===================================================================
# ===============  SECTION 1 — RUN A NEW EXPERIMENT  =================
# ===================================================================
st.header("⚙️ Run a New Experiment")

# ---------------- Device Selector -----------------
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

# ---------------- Model Selection -----------------
model_type = st.selectbox("Choose model type:", ["MLP", "CNN"])

if model_type == "MLP":
    dims_text = st.text_area(
        "MLP hidden dimensions (1 model per line)",
        value="12\n24,4"
    )
else:
    dims_text = st.text_area(
        "CNN channels (1 model per line)",
        value="4\n16"
    )

def parse_list_input(raw_text):
    parsed = []
    for line in raw_text.split("\n"):
        if line.strip():
            parsed.append([int(x.strip()) for x in line.split(",")])
    return parsed

hyperparams = parse_list_input(dims_text)

nbr_epochs = st.number_input("Epochs", min_value=1, value=10)

# ---------------- Run Experiment Button -----------------
run_expe = st.button("🚀 Run Experiment")

if run_expe:
    st.info("Training in progress...")

    progress_bar = st.progress(0)
    progress_text = st.empty()
    if model_type == "MLP":
        res_epochs, summaries, models, labels = expe_simple_MLP(
            hyperparams, train_loader, test_loader, device, nbr_epochs,
            streamlit_progress=progress_bar, streamlit_text=progress_text,
            streamlit_run_name="MLP Experiment"
        )
    else:
        res_epochs, summaries, models, labels = expe_simple_CNN(
            hyperparams, train_loader, test_loader, device, nbr_epochs,
            streamlit_progress=progress_bar, streamlit_text=progress_text
        )

    st.success("Training complete!")


# ====================================================================
# ===========  SECTION 2 — VISUALIZE A PAST EXPERIMENT  ==============
# ====================================================================

st.header("🔍 Visualize Experiments")

checkpoint_dir = "./checkpoints"

names_models = []
for folder in os.listdir(checkpoint_dir):
    if 'SimpleMLP' in folder or 'SimpleCNN' in folder:
        names_models.append(folder)
names_models = sorted(names_models)
names_models = sorted(names_models, key=sort_keys_expe)

names_models_clean = []
for name in names_models:
    clean_name = name.split('-')[1]
    device_name = name.split('-')[0].upper()
    clean_name = clean_name.replace("_", " (", 1).replace("_", ",") + f") on {device_name}"
    names_models_clean.append(clean_name)

selected_model_name_clean = st.selectbox("Select a model to visualize", names_models_clean)
selected_model_name = names_models[names_models_clean.index(selected_model_name_clean)]

hidden = selected_model_name.split('_')[1:]
hidden = [int(h) for h in hidden]
if 'SimpleMLP' in selected_model_name:
    model = SimpleMLP(hidden_dims=hidden)
elif 'SimpleCNN' in selected_model_name:
    model = SimpleCNN(hidden_channels=hidden)
else:
    st.error("Unknown model type selected.")
    st.stop()
model_path = os.path.join(checkpoint_dir, selected_model_name, "model.pth")
model.load_state_dict(torch.load(model_path, map_location='cpu'))

fig_examples = plot_examples_classification(model, test_dataset, device, n=5)
st.subheader("Example Predictions")
st.pyplot(fig_examples)

fig_wrong = plot_wrong_classification(model, test_dataset, device, n=5)
st.subheader("Wrong Predictions")
st.pyplot(fig_wrong)


# ====================================================================
# ===========  SECTION 3 — GLOBAL EXPERIMENT OVERVIEW   ==============
# ====================================================================
st.header("📊 Global Experiments Overview")

checkpoint_dir = "./checkpoints"
all_summaries, all_res_epochs = [], []
# Load all summary.json files from checkpoints/
for root, dirs, files in os.walk(checkpoint_dir):
    for file in files:
        if 'SimpleMLP' in root or 'SimpleCNN' in root:
            if file == "summary.json":
                with open(os.path.join(root, file), "r") as f:
                    summary_dict = json.load(f)
                    summary_df = pd.DataFrame([summary_dict])
                    all_summaries.append(summary_df)
            elif file == "res_epochs.json":
                res_epochs = pd.read_json(os.path.join(root, file))
                this_folder = os.path.basename(root)
                # Before the first '-' is the device, everything after is the run_name
                res_epochs['device'] = this_folder.split('-')[0]
                res_epochs['run_name'] = this_folder[len(res_epochs['device'].iloc[0])+1:]
                all_res_epochs.append(res_epochs)

# --- Plot summary scatter ---

st.subheader("🥽 Summary of Experiments")

def palette_symbol_mappings(df):
    palette = {}
    mlp_runs = df[df['run_name'].str.contains("SimpleMLP")]["run_name"].unique().tolist()
    cnn_runs = df[df['run_name'].str.contains("SimpleCNN")]["run_name"].unique().tolist()
    for i, name in enumerate(mlp_runs):
        r, g, b = sns.color_palette("autumn", len(mlp_runs)).pop(i)
        palette[name] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    for i, name in enumerate(cnn_runs):
        r, g, b = sns.color_palette("winter", len(cnn_runs)).pop(i)
        palette[name] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    for name in df['run_name'].unique():
        if name not in palette:
            palette[name] = "grey"
    
    symbols = {"CPU": "circle", "MPS": "triangle-up", "GPU": "square"}
    for dev in df['device'].unique():
        if dev not in symbols:
            symbols[dev] = "diamond"
    return palette, symbols

def clean_df(df):
    df['device'] = df['device'].str.upper()
    if 'epochs' in df.columns:
        df = df.sort_values(
            by=["run_name", "epochs"],
            key=lambda col: col.map(sort_keys_expe) if col.name == "run_name" else col
        )
    else:
        df = df.sort_values(
            by="run_name",
            key=lambda col: col.map(sort_keys_expe)
        )
    # Replace the first '_' with ' (', the other '_' with ',', and add a closing ')'
    df['run_name'] = df['run_name'].str.replace("_", " (", n=1).str.replace("_", ",") + ")"
    return df


if len(all_summaries) == 0:
    st.warning("No experiments found in checkpoints/ yet.")
else:
    st.write("Loaded experiments:", len(all_summaries))
    summaries = pd.concat(all_summaries, ignore_index=True)
    summaries = clean_df(summaries)
    
    palette, symbols = palette_symbol_mappings(summaries)

    # Metric Selection
    numeric_cols = summaries.select_dtypes(include='number').columns.tolist()

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
        'total_water_L': "Total Water Usage (L)",
        'ram_total_size': "Total RAM Size (GB)"
    }
    label_to_col = {v: k for k, v in labels.items()}

    col1, col2 = st.columns(2)
    x_label = col1.selectbox(
        "X-axis",
        options=[labels[c] for c in numeric_cols],
        index=[labels[c] for c in numeric_cols].index(labels["nbr_parameters"])
    )
    y_label = col2.selectbox(
        "Y-axis",
        options=[labels[c] for c in numeric_cols],
        index=[labels[c] for c in numeric_cols].index(labels["total_emissions_kgCO2eq"])
    )
    x_col = label_to_col[x_label]
    y_col = label_to_col[y_label]
    
    # Plot Interactive Scatter
    fig = px.scatter(
        summaries,
        x=x_col,
        y=y_col,
        color="run_name",
        symbol="device",
        hover_data=summaries.columns,
        color_discrete_map=palette,
        symbol_map=symbols,
        size=[1]*summaries.shape[0],
        size_max=8,
        labels=labels,
        title=f"{x_label} vs {y_label}"
    )

    st.plotly_chart(fig, width='stretch')

# --- Plot res_epochs overview ---

st.subheader("📈 Experiments Over Epochs")

if len(all_res_epochs) == 0:
    st.warning("No epoch results found in checkpoints/ yet.")
else:
    st.write("Loaded epoch results for experiments:", len(all_res_epochs))
    res_epochs_concat = pd.concat(all_res_epochs, ignore_index=True)
    res_epochs_concat = clean_df(res_epochs_concat)
    
    cumul_cols = [
        'train_emissions_kgCO2eq', 'eval_emissions_kgCO2eq',
        'train_total_energy_kWh', 'eval_total_energy_kWh',
        'train_cpu_energy_kWh', 'eval_cpu_energy_kWh',
        'train_gpu_energy_kWh', 'eval_gpu_energy_kWh',
        'train_ram_energy_kWh', 'eval_ram_energy_kWh',
        'train_water_L', 'eval_water_L'
    ]
    for col in cumul_cols:
        res_epochs_concat[f'cumul_{col}'] = res_epochs_concat.groupby('run_name')[col].cumsum()
    
    palette, symbols = palette_symbol_mappings(res_epochs_concat)

    # Metric Selection
    numeric_cols_epochs = res_epochs_concat.select_dtypes(include='number').columns.tolist()
    numeric_cols_epochs.remove('epochs')
    
    labels = {
        "epochs": "Epochs",
        "train_losses": "Train Losses",
        "test_losses": "Test Losses",
        "train_accuracies": "Train Accuracies",
        "test_accuracies": "Test Accuracies",
        "train_emissions_kgCO2eq": "Train Emissions (kgCO2eq)",
        "train_total_energy_kWh": "Train Total Energy (kWh)",
        "train_cpu_energy_kWh": "Train CPU Energy (kWh)",
        "train_gpu_energy_kWh": "Train GPU Energy (kWh)",
        "train_ram_energy_kWh": "Train RAM Energy (kWh)",
        "train_cpu_power_W": "Train CPU Power (W)",
        "train_gpu_power_W": "Train GPU Power (W)",
        "train_ram_power_W": "Train RAM Power (W)",
        "train_water_L": "Train Water Usage (L)",
        "eval_emissions_kgCO2eq": "Eval Emissions (kgCO2eq)",
        "eval_total_energy_kWh": "Eval Total Energy (kWh)",
        "eval_cpu_energy_kWh": "Eval CPU Energy (kWh)",
        "eval_gpu_energy_kWh": "Eval GPU Energy (kWh)",
        "eval_ram_energy_kWh": "Eval RAM Energy (kWh)",
        "eval_cpu_power_W": "Eval CPU Power (W)",
        "eval_gpu_power_W": "Eval GPU Power (W)",
        "eval_ram_power_W": "Eval RAM Power (W)",
        "eval_water_L": "Eval Water Usage (L)",
        "device": "Device",
        "run_name": "Experiment"
    }
    for col in cumul_cols:
        labels[f'cumul_{col}'] = f"Cumulative {labels[col]}"
    label_to_col = {v: k for k, v in labels.items()}

    col1, col2 = st.columns(2)
    y_label_epoch = col1.selectbox(
        "Y-axis",
        options=[labels[c] for c in numeric_cols_epochs],
        index=[labels[c] for c in numeric_cols_epochs].index(labels["test_accuracies"])
    )
    y_col_epoch = label_to_col[y_label_epoch]

    fig_epoch = go.Figure()

    # Iterate over each unique run
    for run in res_epochs_concat['run_name'].unique():
        for device in res_epochs_concat['device'].unique():
            df_run = res_epochs_concat[
                (res_epochs_concat['run_name'] == run) & 
                (res_epochs_concat['device'] == device)
            ]
            color = palette.get(run, None)

            fig_epoch.add_trace(
                go.Scatter(
                    x=df_run['epochs'],
                    y=df_run[y_col_epoch],
                    mode='lines+markers',
                    name=f"{run} ({device})",
                    line=dict(color=color, width=1),
                    marker=dict(symbol=symbols.get(device, "circle"), size=8),
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Epoch: %{x}<br>" +
                        f"{y_label_epoch}: %{{y}}<br>" +
                        "Device: %{customdata}"
                    ),
                    text=[run]*len(df_run),
                    customdata=df_run['device']
                )
            )

    fig_epoch.update_layout(
        title=f"{y_label_epoch} over Epochs",
        xaxis_title="Epochs",
        yaxis_title=y_label_epoch,
        legend_title="Run Name",
        template="plotly_white"
    )

    st.plotly_chart(fig_epoch, width='stretch')


