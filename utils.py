from tqdm import tqdm
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------
# TRAINING AND EVALUATION UTILITIES
# ------------------------------------

def train_epoch(model, train_loader, optimizer, criterion, device, progress_bar=True):
    model.train()
    if progress_bar:
        iterator = tqdm(train_loader, ncols=80, desc="Loop over batches")
    else:
        iterator = train_loader
    for data, target in iterator:
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

def full_train(model, train_loader, test_loader, criterion, optimizer, nbr_epochs=5, device='cpu', verbose=True):

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    if verbose:
        print("Initial evaluation: ", end="")
    initial_train_loss, initial_train_accuracy = evaluate(model, train_loader, criterion, device)
    initial_test_loss, initial_test_accuracy = evaluate(model, test_loader, criterion, device)
    if verbose:
        print(f"Train Loss: {initial_train_loss:.4f}, Train Acc: {initial_train_accuracy:.2f}%, "
              f"Test Loss: {initial_test_loss:.4f}, Test Acc: {initial_test_accuracy:.2f}%")
    train_losses.append(initial_train_loss)
    test_losses.append(initial_test_loss)
    train_accuracies.append(initial_train_accuracy)
    test_accuracies.append(initial_test_accuracy)

    if verbose:
        iterator = range(nbr_epochs)
    else:
        iterator = tqdm(range(nbr_epochs), ncols=50, desc="Epochs")
    for epoch in iterator:
        if verbose:
            print(f"  > Epoch {epoch+1}/{nbr_epochs}")
        # Train
        train_epoch(model, train_loader, optimizer, criterion, device, progress_bar=verbose)

        # Evaluate
        if verbose:
            print("Evaluation after epoch: ", end="")
        train_loss, train_accuracy = evaluate(model, train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        if verbose:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        # Log the variables
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    
    output = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies
    }
    
    return output

# ------------------------------------
# EXPERIMENT UTILITIES
# ------------------------------------

def experiment_classif_simple(model, train_loader, test_loader, nbr_epochs=5, lr=0.001, run_name="", verbose=False, device='cpu'):
    # Training the model
    nbr_parameters = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Number of parameters: {nbr_parameters}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    t_start = datetime.now()
    output = full_train(
        model, train_loader, test_loader, criterion, optimizer, 
        nbr_epochs=nbr_epochs, device=device, verbose=verbose
    )
    total_training_time = datetime.now() - t_start
    if verbose:
        print(f"Total training time: {total_training_time}")

    output["nbr_parameters"] = nbr_parameters
    output["total_training_time_sec"] = total_training_time.total_seconds()
    output["run_name"] = run_name
    output['device'] = device
    
    return output

# ------------------------------------
# RESULTS FORMATTING
# ------------------------------------

def format_results(all_outputs):

    # Build full wide dataframe first
    results_df = pd.concat(
        [pd.DataFrame(o).assign(epochs=range(len(o["train_losses"])))
         for o in all_outputs],
        ignore_index=True
    )

    # ---- Accuracy long block ----
    acc_long = results_df.melt(
        id_vars=["epochs", "run_name"],
        value_vars=["train_accuracies", "test_accuracies"],
        var_name="accuracy_type",
        value_name="accuracy_value"
    )

    # ---- Loss long block ----
    loss_long = results_df.melt(
        id_vars=["epochs", "run_name"],
        value_vars=["train_losses", "test_losses"],
        var_name="loss_type",
        value_name="loss_value"
    )

    # ---- Emission long block ----
    if "train_emissions" in results_df.columns and "eval_emissions" in results_df.columns:
        emission_long = results_df.melt(
            id_vars=["epochs", "run_name"],
            value_vars=["train_emissions", "eval_emissions"],
            var_name="emission_type",
            value_name="emission_value"
        )

    # Merge accuracy and loss side-by-side
    results_long = acc_long.merge(
        loss_long,
        on=["epochs", "run_name"],
        how="left"
    )
    if "train_emissions" in results_df.columns and "eval_emissions" in results_df.columns:
        results_long = results_long.merge(
            emission_long,
            on=["epochs", "run_name"],
            how="left"
        )

    # Summary information
    results_summary = results_df[[
        cols for cols in results_df.columns if cols not in
        ["train_accuracies", "test_accuracies", "train_losses", "test_losses", 
         "train_emissions", "eval_emissions", "epochs"]
    ]].drop_duplicates()

    return results_long, results_summary

# ------------------------------------
# PLOTTING UTILITIES
# ------------------------------------

def plot_examples_classification(model, data_loader, device, n=5):
    model.eval()
    data_iter = next(iter(data_loader))
    images, labels = data_iter[0][:n], data_iter[1][:n]
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}, True: {labels[i].item()}")
        plt.axis('off')
    plt.show()

def plot_loss_acc_over_epochs(results_long):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    for val, lab, j in [('train', 'Train', 0), ('test', 'Test', 1)]:
        # Loss plots
        sns.lineplot(
            ax=axes[0, j], data=results_long[results_long["loss_type"] == val+'_losses'], 
            x="epochs", y="loss_value", marker='o', hue="run_name"
        )
        axes[0, j].set_xlabel("Epoch")
        axes[0, j].set_ylabel(f"{lab} Loss")
        axes[0, j].set_yscale("log")
        axes[0, j].legend(title="Model Architecture")
        axes[0, j].grid(visible=True)

        # Accuracy plots
        sns.lineplot(
            ax=axes[1, j], data=results_long[results_long["accuracy_type"] == val+'_accuracies'], 
            x="epochs", y="accuracy_value", marker='o', hue="run_name"
        )
        axes[1, j].set_xlabel("Epoch")
        axes[1, j].set_ylabel(f"{lab} Accuracy (%)")
        axes[1, j].legend(title="Model Architecture")
        axes[1, j].grid(visible=True)

    plt.suptitle("Training and Test Loss/Accuracy over Epochs", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_time_vs_parameters(results_summary):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=results_summary, x="nbr_parameters", y="total_training_time_sec", hue="run_name")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Total Training Time (seconds)")
    plt.title("Training Time vs Number of Parameters")
    plt.legend(title="Model")
    plt.grid()
    plt.show()
