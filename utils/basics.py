from tqdm import tqdm
import re
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA

from codecarbon import EmissionsTracker

from utils.models import CVAE_CNN, CVAE_MLP


# ------------------------------------
# EMISSIONS TRACKING UTILITIES
# ------------------------------------

def get_impact_task(emission, verbose=False):
    out = {
        "emissions_kgCO2eq": emission.emissions,
        "total_energy_kWh": emission.energy_consumed,
        "cpu_energy_kWh": emission.cpu_energy,
        "gpu_energy_kWh": emission.gpu_energy,
        "ram_energy_kWh": emission.ram_energy,
        "cpu_power_W": emission.cpu_power,
        "gpu_power_W": emission.gpu_power,
        "ram_power_W": emission.ram_power,
        "water_L": emission.water_consumed,
    }
    if verbose:
        for k, v in out.items():
            print(f"{k}: {v}")
    return out

def get_impact_tracker(tracker, verbose=False):
    out = {
        "total_emissions_kgCO2eq": tracker.final_emissions,
        "total_energy_kWh": tracker._total_energy.kWh,
        "total_cpu_energy_kWh": tracker._total_cpu_energy.kWh,
        "total_gpu_energy_kWh": tracker._total_gpu_energy.kWh,
        "total_ram_energy_kWh": tracker._total_ram_energy.kWh,
        "cpu_power_W": tracker._cpu_power.W,
        "gpu_power_W": tracker._gpu_power.W,
        "ram_power_W": tracker._ram_power.W,
        "total_water_L": tracker._total_water.litres,
        "cpu_model": tracker.final_emissions_data.cpu_model,
        "gpu_model": tracker.final_emissions_data.gpu_model,
        "ram_total_size": tracker.final_emissions_data.ram_total_size
    }
    if verbose:
        for k, v in out.items():
            print(f"{k}: {v}")
    return out

def print_emission_streamlit(emission_in):

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
        else:
            emission[key] = str(emission[key])
        if key not in labels:
            labels[key] = key
    emission = pd.DataFrame.from_dict(emission, orient='index', columns=['Value'])
    emission.index = emission.index.map(labels)
    st.table(emission)

# ------------------------------------
# TRAINING AND EVALUATION UTILITIES
# ------------------------------------

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, data_loader, criterion, device):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss = loss / len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)
    return loss, accuracy

def full_train(model, train_loader, test_loader, criterion, optimizer, 
               nbr_epochs=5, device='cpu', tracker=None, 
               streamlit_progress=None, streamlit_text=None, streamlit_run_name=None,
               streamlit_n_expe=None, streamlit_i_current_expe=None):
    # Setup metrics logging
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    # Initial evaluation (eventually with emissions tracking)
    if tracker is not None:
        output_emissions_epochs_train, output_emissions_epochs_eval = [], []
        tracker.start_task(task_name="Eval epoch 0")
    initial_train_loss, initial_train_accuracy = evaluate(model, train_loader, criterion, device)
    initial_test_loss, initial_test_accuracy = evaluate(model, test_loader, criterion, device)
    if tracker is not None:
        tmp_emissions = tracker.stop_task()
        output_emissions_epochs_eval.append(get_impact_task(tmp_emissions, verbose=False))
        output_emissions_epochs_train.append({k: 0.0 for k in output_emissions_epochs_eval[-1].keys()})
    # Log initial metrics
    train_losses.append(initial_train_loss)
    test_losses.append(initial_test_loss)
    train_accuracies.append(initial_train_accuracy)
    test_accuracies.append(initial_test_accuracy)
    # Loop over epochs
    for epoch in tqdm(range(nbr_epochs), ncols=50, desc="Epochs"):
        # Train
        if tracker is not None:
            tracker.start_task(task_name=f"Train epoch {epoch+1}")
        train_epoch(model, train_loader, optimizer, criterion, device)
        if tracker is not None:
            tmp_emissions = tracker.stop_task()
            output_emissions_epochs_train.append(get_impact_task(tmp_emissions, verbose=False))
        # Evaluate
        if tracker is not None:
            tracker.start_task(task_name=f"Eval epoch {epoch+1}")
        train_loss, train_accuracy = evaluate(model, train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        if tracker is not None:
            tmp_emissions = tracker.stop_task()
            output_emissions_epochs_eval.append(get_impact_task(tmp_emissions, verbose=False))
        # Log metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if streamlit_progress is not None:
            streamlit_progress.progress(
                (epoch + 1) / nbr_epochs / streamlit_n_expe
                + (streamlit_i_current_expe / streamlit_n_expe)
            )
        if streamlit_text is not None:
            streamlit_text.text(
                f"Running experiment: {streamlit_run_name}\n" +
                f"Epoch {epoch+1}/{nbr_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
            )
    # Format output dataframe
    output = pd.DataFrame({
        "epochs": range(nbr_epochs + 1),
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies
    })
    if tracker is not None:
        output = output.merge(
            pd.DataFrame(output_emissions_epochs_train)
            .assign(epochs=range(nbr_epochs + 1))
            .add_prefix("train_"),
            left_on="epochs", right_on="train_epochs"
        )
        output = output.merge(
            pd.DataFrame(output_emissions_epochs_eval)
            .assign(epochs=range(nbr_epochs + 1))
            .add_prefix("eval_"),
            left_on="epochs", right_on="eval_epochs"
        )
        output.drop(columns=["train_epochs", "eval_epochs"], inplace=True)
    return output

# ------------------------------------
# EXPERIMENT UTILITIES
# ------------------------------------

def experiment_classif_simple(model, train_loader, test_loader, nbr_epochs=5, lr=0.001, 
                              run_name="", device='cpu', track_emissions=False, 
                              streamlit_progress=None, streamlit_text=None, streamlit_run_name=None, 
                              streamlit_n_expe=None, streamlit_i_current_expe=None):
    # Count parameters
    nbr_parameters = sum(p.numel() for p in model.parameters())
    # Setup
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Track time
    t_start = datetime.now()
    # Setup emissions tracker
    if track_emissions:
        tracker = EmissionsTracker(save_to_file=False, log_level='error')
    else:
        tracker = None
    # Training
    df_res_epochs = full_train(
        model, train_loader, test_loader, criterion, optimizer, 
        nbr_epochs=nbr_epochs, device=device, tracker=tracker,
        streamlit_progress=streamlit_progress, streamlit_text=streamlit_text,
        streamlit_run_name=streamlit_run_name, streamlit_n_expe=streamlit_n_expe,
        streamlit_i_current_expe=streamlit_i_current_expe
    )
    # Log emissions from tracker
    if track_emissions:
        _ = tracker.stop()
        emission_info = get_impact_tracker(tracker, verbose=False)
    # Get total time
    total_training_time = datetime.now() - t_start
    # Log output summary
    output_summary = {
        "run_name": run_name,
        "nbr_parameters": nbr_parameters,
        "nbr_epochs": nbr_epochs,
        "device": str(device),
        "total_training_time_sec": total_training_time.total_seconds(),
        "final_train_loss": df_res_epochs["train_losses"].iloc[-1],
        "final_test_loss": df_res_epochs["test_losses"].iloc[-1],
        "final_train_accuracy": df_res_epochs["train_accuracies"].iloc[-1],
        "final_test_accuracy": df_res_epochs["test_accuracies"].iloc[-1]
    }
    if track_emissions:
        output_summary.update(emission_info)
    
    return df_res_epochs, output_summary


def sort_keys_expe(name):
    # 1) Model family priority
    if "SimpleMLP" in name:
        family = 0
    elif "SimpleCNN" in name:
        family = 1
    else:
        family = 2   # Everything else
    # 2) Split by "_"
    parts = name.split("_")
    underscore_count = len(parts) - 1 # Count of "_" defines complexity of the model
    # 3) Extract numeric suffixes for numeric sorting
    numeric_parts = []
    for p in parts[1:]:  # skip the "SimpleMLP" or "SimpleCNN"
        if p.isdigit():
            numeric_parts.append(int(p))
        else:
            numeric_parts.append(float("inf")) # fallback if not numeric
    return (family, underscore_count, *numeric_parts)


# ------------------------------------
# PLOTTING UTILITIES
# ------------------------------------

def plot_examples_classification(model, dataset, device, n=5, suptitle=None):
    model.to(device)
    model.eval()
    idxs = torch.randperm(len(dataset))[:n]
    images = []
    labels = []
    for idx in idxs:
        img, lbl = dataset[idx]
        images.append(img)
        labels.append(lbl)
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(n*2, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}, True: {labels[i].item()}")
        plt.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig

def plot_wrong_classification(model, dataset, device, n=5, max_attempts=10000, suptitle=None):
    model.eval()
    # Search for wrong classifications
    wrong = []
    attempts = 0
    N = len(dataset)
    while len(wrong) < n and attempts < max_attempts:
        idx = np.random.randint(0, N)
        img, lbl = dataset[idx]
        img_gpu = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_gpu).argmax(dim=1).item()
        if pred != lbl:
            wrong.append((img, lbl, pred))
        attempts += 1

    if len(wrong) == 0:
        print("No wrong classifications found (within attempts).")
        return
    elif len(wrong) < n:
        print(f"Only found {len(wrong)} wrong classifications (within attempts).")
        n = len(wrong)
    fig = plt.figure(figsize=(n*2, 3))
    for i, (img, lbl, pred) in enumerate(wrong):
        plt.subplot(1, n, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Pred: {pred}, True: {lbl}")
        plt.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig

def plot_loss_acc_over_epochs(all_res_epochs, all_labels, palette):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for df, label in zip(all_res_epochs, all_labels):
        for j, split in enumerate(['train', 'test']):
            # Loss plots
            ax = axes[0, j]
            ax.plot(df['epochs'], df[f'{split}_losses'], marker='o', label=label, c=palette[label])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"{split.capitalize()} Loss")
            ax.set_yscale("log")
            ax.legend(title="Model Architecture", fontsize='small')
            ax.grid(visible=True)
            # Accuracy plots
            ax = axes[1, j]
            ax.plot(df['epochs'], df[f'{split}_accuracies'], marker='o', label=label, c=palette[label])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"{split.capitalize()} Accuracy (%)")
            ax.legend(title="Model Architecture", fontsize='small')
            ax.grid(visible=True)

    plt.suptitle("Training and Test Loss/Accuracy over Epochs", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_co2_energy_over_epochs(all_res_epochs, all_labels, palette):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for df, label in zip(all_res_epochs, all_labels):
        # Emissions
        ax = axes[0, 0]
        ax.plot(df['epochs'], (df['train_emissions_kgCO2eq']+df['eval_emissions_kgCO2eq']).cumsum(), 
                marker='o', label=label, c=palette[label])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Emissions (kg CO2eq)")
        ax.legend(title="Model Architecture", fontsize='small')
        ax.grid(visible=True)
        # RAM energy
        ax = axes[0, 1]
        ax.plot(df['epochs'], (df['train_ram_energy_kWh']+df['eval_ram_energy_kWh']).cumsum(), 
                marker='o', label=label, c=palette[label])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total RAM Energy (kWh)")
        ax.legend(title="Model Architecture", fontsize='small')
        ax.grid(visible=True)
        # CPU energy
        ax = axes[1, 0]
        ax.plot(df['epochs'], (df['train_cpu_energy_kWh']+df['eval_cpu_energy_kWh']).cumsum(), 
                marker='o', label=label, c=palette[label])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total CPU Energy (kWh)")
        ax.legend(title="Model Architecture", fontsize='small')
        ax.grid(visible=True)
        # GPU energy
        ax = axes[1, 1]
        ax.plot(df['epochs'], (df['train_gpu_energy_kWh']+df['eval_gpu_energy_kWh']).cumsum(), 
                marker='o', label=label, c=palette[label])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total GPU Energy (kWh)")
        ax.legend(title="Model Architecture", fontsize='small')
        ax.grid(visible=True)

    plt.suptitle("Cumulative Emissions and Energy Consumption over Epochs", y=1.02)
    plt.tight_layout()
    plt.show()

# ------------------------------------
# Conditional VAE Models
# ------------------------------------

def vae_loss(x_hat, x, mu, logvar):
    B = x.size(0)
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + kl) / B

def train_cvae(model, dataloader, epochs=10, lr=1e-3, device="cpu", 
               track_emissions=False, streamlit_progress=None, streamlit_text=None):
    # Setup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Setup emissions tracker
    if track_emissions:
        tracker = EmissionsTracker(save_to_file=False, log_level='error')
        tracker.start()
    # Track time
    t_start = datetime.now()
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            # Load data
            x = x.to(device)
            y = y.to(device)
            y_onehot = F.one_hot(y, num_classes=model.num_classes).float().to(device)
            # Forward + backward + optimize
            optimizer.zero_grad()
            if isinstance(model, CVAE_CNN):
                x_hat, mu, logvar = model(x, y)
            elif isinstance(model, CVAE_MLP):
                x_hat, mu, logvar = model(x, y_onehot)
            loss = vae_loss(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()
            # Log loss
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | loss = {total_loss / len(dataloader.dataset):.4f}")
        if streamlit_progress is not None:
            streamlit_progress.progress((epoch + 1) / epochs)
        if streamlit_text is not None:
            streamlit_text.text(
                f"Epoch {epoch+1}/{epochs} | loss = {total_loss / len(dataloader.dataset):.4f}"
            )
    # Track time
    total_training_time = datetime.now() - t_start
    print(f"Total training time: {total_training_time}")
    # Log emissions from tracker
    if track_emissions:
        _ = tracker.stop()
        emission_info = get_impact_tracker(tracker, verbose=False)
        emission_info["total_training_time_sec"] = total_training_time.total_seconds()
        return emission_info

def generate_digit_cvae(model, n_samples=8, device="cpu", track_emissions=False, plot=True):
    # Setup emissions tracker
    if track_emissions:
        tracker = EmissionsTracker(save_to_file=False, log_level='error')
        tracker.start()
    # Generate n_samples digits for each class
    all_imgs = []
    model.to(device)
    model.eval()
    for digit in range(10):
        with torch.no_grad():
            y = torch.full((n_samples,), digit, dtype=torch.long).to(device)
            y_onehot = F.one_hot(y, num_classes=10).float()
            z = torch.randn(n_samples, model.latent_dim).to(device)
            if isinstance(model, CVAE_CNN):
                imgs = model.decode(z, y).cpu()
            elif isinstance(model, CVAE_MLP):
                imgs = model.decode(z, y_onehot).cpu()
            all_imgs.append(imgs)
    # Log emissions from tracker
    if track_emissions:
        _ = tracker.stop()
        emission_info = get_impact_tracker(tracker, verbose=False)
    # Plot generated digits
    if plot:
        _ = plot_generate_digit(all_imgs)
        plt.show()

    if track_emissions:
        return all_imgs, emission_info
    else:
        return all_imgs
    
def plot_generate_digit(all_imgs):
    digits = len(all_imgs)
    n_samples = max(imgs.shape[0] for imgs in all_imgs)
    scale = 0.7
    fig = plt.figure(figsize=(n_samples*scale, digits*scale))
    for digit in range(digits):
        imgs = all_imgs[digit]
        for i in range(n_samples):
            plt.subplot(digits, n_samples, digit*n_samples + i + 1)
            if i < imgs.shape[0]:
                plt.imshow(imgs[i].squeeze(), cmap="gray")
            plt.axis("off")
    return fig

# ------------------------------------
# OTHER SIMPLE GENERATION MODELS
# ------------------------------------

def train_PCA(X_train, y_train, n_components=50, track_emissions=False):
    if track_emissions:
        tracker = EmissionsTracker(save_to_file=False, log_level='error')
        tracker.start()

    all_pcas = {}
    for digit in range(10):
        X_digit = X_train[y_train == digit]
        pca = PCA(n_components=n_components)
        pca.fit(X_digit)
        all_pcas[digit] = pca

    if track_emissions:
        _ = tracker.stop()
        emission_info = get_impact_tracker(tracker, verbose=False)
        return all_pcas, emission_info
    else:
        return all_pcas

def generate_digit_pca(all_pcas, n_samples=8, track_emissions=False, plot=True):
    if track_emissions:
        tracker = EmissionsTracker(save_to_file=False, log_level='error')
        tracker.start()
    all_imgs = []
    for digit in range(10):
        pca = all_pcas[digit]
        n_components = pca.n_components_
        # Sample from standard normal in PCA space
        coeffs = np.random.normal(0, 1, size=(n_samples, n_components))
        imgs = pca.inverse_transform(coeffs).reshape(n_samples, 28, 28)
        all_imgs.append(imgs)
    if track_emissions:
        _ = tracker.stop()
        emission_info = get_impact_tracker(tracker, verbose=False)
    # Plot generated digits
    if plot:
        _ = plot_generate_digit(all_imgs)
        plt.show()
    
    if track_emissions:
        return all_imgs, emission_info
    else:
        return all_imgs

def train_proba_pixel(X_train, y_train, track_emissions=False):
    if track_emissions:
        tracker = EmissionsTracker(save_to_file=False, log_level='error')
        tracker.start()
    all_p = {}
    for digit in range(10):
        X_digit = X_train[y_train == digit] / 255.0
        # pixel probability
        p = X_digit.mean(axis=0)
        all_p[digit] = p
    if track_emissions:
        _ = tracker.stop()
        emission_info = get_impact_tracker(tracker, verbose=False)
        return all_p, emission_info
    else:
        return all_p

def generate_digit_proba_pixel(all_p, n_samples=8, track_emissions=False, plot=True):
    if track_emissions:
        tracker = EmissionsTracker(save_to_file=False, log_level='error')
        tracker.start()
    all_imgs = []
    for digit in range(10):
        p = all_p[digit]
        imgs = np.random.binomial(1, p, size=(n_samples, 28*28)).reshape(n_samples, 28, 28)
        all_imgs.append(imgs)
    if track_emissions:
        _ = tracker.stop()
        emission_info = get_impact_tracker(tracker, verbose=False)
    # Plot generated digits
    if plot:
        _ = plot_generate_digit(all_imgs)
        plt.show()
    
    if track_emissions:
        return all_imgs, emission_info
    else:
        return all_imgs
