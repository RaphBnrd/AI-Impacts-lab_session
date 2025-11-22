from tqdm import tqdm
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

from models import CVAE_CNN, CVAE_MLP


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
               nbr_epochs=5, device='cpu', tracker=None):
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
                              run_name="", device='cpu', track_emissions=False):
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
        nbr_epochs=nbr_epochs, device=device, tracker=tracker
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

# ------------------------------------
# PLOTTING UTILITIES
# ------------------------------------

def plot_examples_classification(model, data_loader, device, n=5, suptitle=None):
    model.to(device)
    model.eval()
    data_iter = next(iter(data_loader))
    images, labels = data_iter[0][:n], data_iter[1][:n]
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(n*2, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}, True: {labels[i].item()}")
        plt.axis('off')
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

def plot_wrong_classification(model, data_loader, device, n=5, suptitle=None):
    model.eval()
    wrong_images = []
    wrong_labels = []
    wrong_preds = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    wrong_images.append(images[i].cpu())
                    wrong_labels.append(labels[i].item())
                    wrong_preds.append(preds[i].item())
                if len(wrong_images) >= n:
                    break
            if len(wrong_images) >= n:
                break
    if len(wrong_images) == 0:
        print("No wrong classifications found.")
        return
    elif len(wrong_images) < n:
        n = len(wrong_images)
    plt.figure(figsize=(n*2, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(wrong_images[i].squeeze(), cmap='gray')
        plt.title(f"Pred: {wrong_preds[i]}, True: {wrong_labels[i]}")
        plt.axis('off')
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

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
               track_emissions=False):
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
    # Track time
    total_training_time = datetime.now() - t_start
    print(f"Total training time: {total_training_time}")
    # Log emissions from tracker
    if track_emissions:
        _ = tracker.stop()
        emission_info = get_impact_tracker(tracker, verbose=False)
        emission_info["total_training_time_sec"] = total_training_time.total_seconds()
        return emission_info

def generate_digit_cvae(model, n_samples=8, device="cpu", track_emissions=False):
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
    scale = 0.7
    plt.figure(figsize=(n_samples*scale, 10*scale))
    for digit in range(10):
        imgs = all_imgs[digit]
        for i in range(n_samples):
            plt.subplot(10,n_samples,digit*n_samples+i+1)
            plt.imshow(imgs[i].squeeze(), cmap="gray")
            plt.axis("off")
    plt.show()

    if track_emissions:
        return imgs, emission_info
    else:
        return imgs

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

def generate_digit_pca(all_pcas, n_samples=8, track_emissions=False):
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
    # Plot
    scale = 0.7
    plt.figure(figsize=(n_samples*scale, 10*scale))
    for digit in range(10):
        imgs = all_imgs[digit]
        for i in range(n_samples):
            plt.subplot(10, n_samples, digit*n_samples + i + 1)
            plt.imshow(imgs[i], cmap='gray')
            plt.axis('off')
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

def generate_digit_proba_pixel(all_p, n_samples=8, track_emissions=False):
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
    # Plot
    scale=0.7
    plt.figure(figsize=(n_samples*scale, 10*scale))
    for digit in range(10):
        imgs = all_imgs[digit]
        for i in range(n_samples):
            plt.subplot(10, n_samples, digit*n_samples + i + 1)
            plt.imshow(imgs[i], cmap='gray')
            plt.axis('off')
    plt.show()
    if track_emissions:
        return all_imgs, emission_info
    else:
        return all_imgs
