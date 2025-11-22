import os
import json
import pandas as pd
import torch

from utils import experiment_classif_simple, train_cvae

from models import SimpleMLP, SimpleCNN, CVAE_MLP, CVAE_CNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from codecarbon import EmissionsTracker



def import_MNIST(batch_size_train=64):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Plot examples of digits (with true labels)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(example_data[i].cpu().squeeze(), cmap='gray')
        plt.title(f"True: {example_targets[i].item()}")
        plt.axis("off")
    plt.tight_layout()

    return train_loader, test_loader, train_dataset, test_dataset


def expe_simple_MLP(hidden_dims_tested, train_loader, test_loader,
                    device, nbr_epochs=10):
    all_res_epochs_mlp = []
    all_summaries_mlp = []
    all_models_mlp = []
    all_labels_mlp = []

    for hidden_dims in hidden_dims_tested:
        print(f"> Training SimpleMLP with hidden dimensions: {hidden_dims}")
        
        label = f"SimpleMLP_{'_'.join(map(str, hidden_dims))}"
        folder_path = f'checkpoints/{str(device)}-{label}/'
        
        # If the log already exists, skip training and load the model
        if os.path.exists(f'{folder_path}res_epochs.json'):
            print(f"Skipping training {label} as log already exists.")
            # Load the model and results
            df_res_epochs = pd.read_json(f'{folder_path}res_epochs.json')
            with open(f'{folder_path}summary.json', 'r') as f:
                output_summary = json.load(f)
            model = SimpleMLP(hidden_dims=hidden_dims)
            model.load_state_dict(torch.load(f'{folder_path}model.pth', map_location=device))
        else:
            # Train the model
            model = SimpleMLP(hidden_dims=hidden_dims)
            df_res_epochs, output_summary = experiment_classif_simple(
                model, train_loader, test_loader,
                nbr_epochs=nbr_epochs, device=device, run_name=label,
                track_emissions=True
            )
            # Save the model and results
            os.makedirs(folder_path, exist_ok=True)
            df_res_epochs.to_json(f'{folder_path}res_epochs.json')
            with open(f'{folder_path}summary.json', 'w') as f:
                json.dump(output_summary, f)
            torch.save(model.state_dict(), f'{folder_path}model.pth')
        
        all_res_epochs_mlp.append(df_res_epochs)
        all_summaries_mlp.append(output_summary)
        all_models_mlp.append(model)
        all_labels_mlp.append(label)

    return all_res_epochs_mlp, all_summaries_mlp, all_models_mlp, all_labels_mlp


def expe_simple_CNN(hidden_channels_tested, train_loader, test_loader,
                    device, nbr_epochs=10):
    all_res_epochs_cnn = []
    all_summaries_cnn = []
    all_models_cnn = []
    all_labels_cnn = []

    for hidden_chans in hidden_channels_tested:
        print(f"> Training SimpleCNN with hidden channels: {hidden_chans}")
        
        label = f"SimpleCNN_{'_'.join(map(str, hidden_chans))}"
        folder_path = f'checkpoints/{str(device)}-{label}/'

        # If the log already exists, skip training and load the model
        if os.path.exists(f'{folder_path}res_epochs.json'):
            print(f"Skipping training {label} as log already exists.")
            df_res_epochs = pd.read_json(f'{folder_path}res_epochs.json')
            with open(f'{folder_path}summary.json', 'r') as f:
                output_summary = json.load(f)
            model = SimpleCNN(hidden_channels=hidden_chans)
            model.load_state_dict(torch.load(f'{folder_path}model.pth', map_location=device))
        # Train the model
        else:
            model = SimpleCNN(hidden_channels=hidden_chans)
            df_res_epochs, output_summary = experiment_classif_simple(
                model, train_loader, test_loader,
                nbr_epochs=nbr_epochs, device=device, run_name=label,
                track_emissions=True
            )
            os.makedirs(folder_path, exist_ok=True)
            df_res_epochs.to_json(f'{folder_path}res_epochs.json')
            with open(f'{folder_path}summary.json', 'w') as f:
                json.dump(output_summary, f)
            torch.save(model.state_dict(), f'{folder_path}model.pth')

        all_res_epochs_cnn.append(df_res_epochs)
        all_summaries_cnn.append(output_summary)
        all_models_cnn.append(model)
        all_labels_cnn.append(label)

    return all_res_epochs_cnn, all_summaries_cnn, all_models_cnn, all_labels_cnn


def train_cvae_MLP(hidden_dims, latent_dim, train_loader, device, nbr_epochs=10):

    cvae_mlp = CVAE_MLP(hidden_dims=hidden_dims, latent_dim=latent_dim)

    num_params = sum(p.numel() for p in cvae_mlp.parameters() if p.requires_grad)
    print(f"CVAE_MLP number of parameters: {num_params}")

    folder_path = f'checkpoints/{str(device)}-CVAE_MLP_{"_".join(map(str, hidden_dims))}_latent{latent_dim}/'

    if os.path.exists(folder_path + 'model.pth'):
        print("Loading pre-trained CVAE_MLP model.")
        # Load model and results
        cvae_mlp.load_state_dict(
            torch.load(folder_path + 'model.pth', map_location=device)
        )
        with open(f'{folder_path}emission_train.json', 'r') as f:
            emission_train_cvae_mlp = json.load(f)
    else:
        emission_train_cvae_mlp = train_cvae(
            cvae_mlp, train_loader, epochs=nbr_epochs, device=device, track_emissions=True
        )
        # Save model and results
        os.makedirs(folder_path, exist_ok=True)
        torch.save(cvae_mlp.state_dict(), f'{folder_path}model.pth')
        with open(f'{folder_path}emission_train.json', 'w') as f:
            json.dump(emission_train_cvae_mlp, f)

    print("--------------------------------")
    print("CVAE_MLP Training Emissions:")
    for k, v in emission_train_cvae_mlp.items():
        if isinstance(v, float):
            message = f"{k}: {v:.3f}" if v >= 1e-2 else f"{k}: {v:.2e}"
        else:
            message = f"{k}: {v}"
        print(message)
    
    return cvae_mlp, emission_train_cvae_mlp


def train_cvae_CNN(hidden_channels, latent_dim, train_loader, device, nbr_epochs=10):

    cvae_cnn = CVAE_CNN(hidden_channels=hidden_channels, latent_dim=latent_dim)

    num_params = sum(p.numel() for p in cvae_cnn.parameters() if p.requires_grad)
    print(f"CVAE_CNN number of parameters: {num_params}")

    folder_path = f'checkpoints/{str(device)}-CVAE_CNN_{"_".join(map(str, hidden_channels))}_latent{latent_dim}/'

    if os.path.exists(folder_path + 'model.pth'):
        print("Loading pre-trained CVAE_CNN model.")
        # Load model and results
        cvae_cnn.load_state_dict(
            torch.load(folder_path + 'model.pth', map_location=device)
        )
        with open(f'{folder_path}emission_train.json', 'r') as f:
            emission_train_cvae_cnn = json.load(f)
    else:
        emission_train_cvae_cnn = train_cvae(
            cvae_cnn, train_loader, epochs=nbr_epochs, device=device, track_emissions=True
        )
        # Save model and results
        os.makedirs(folder_path, exist_ok=True)
        torch.save(cvae_cnn.state_dict(), f'{folder_path}model.pth')
        with open(f'{folder_path}emission_train.json', 'w') as f:
            json.dump(emission_train_cvae_cnn, f)

    print("--------------------------------")
    print("CVAE_CNN Training Emissions:")
    for k, v in emission_train_cvae_cnn.items():
        if isinstance(v, float):
            message = f"{k}: {v:.3f}" if v >= 1e-2 else f"{k}: {v:.2e}"
        else:
            message = f"{k}: {v}"
        print(message)
    
    return cvae_cnn, emission_train_cvae_cnn


def inference_bert(text, model, tokenizer, device, max_length=200):

    tracker_bert = EmissionsTracker(save_to_file=False, log_level='error')
    tracker_bert.start()
    
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Get the token id with the highest probability at the masked position
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = predictions[0, mask_token_index, :].argmax(dim=-1)
    
    top_k = 5
    top_k_token_ids = predictions[0, mask_token_index, :].topk(top_k).indices
    top_k_token_probs = predictions[0, mask_token_index, :].topk(top_k).values
    str_top_k = [f"{tokenizer.decode([token_id])} ({prob.item():.1f})" for token_id, prob in zip(top_k_token_ids[0], top_k_token_probs[0])]

    predicted_token = tokenizer.decode(predicted_token_id)
    
    emission_bert = tracker_bert.stop()
    print("\n--------------------------------\n")
    print(f"Original text: {text}")
    print(f"Predicted token (top {top_k}): {str_top_k}")
    print(f"Filled sentence: {text.replace(tokenizer.mask_token, predicted_token)}")
    print(f"\n  > BERT Prediction Emissions: {emission_bert:.2e} kg CO2eq")

    return predicted_token, emission_bert


def generate_gpt2(prompt, gpt2_model, gpt2_tokenizer, device, max_length=200):

    tracker_gpt2 = EmissionsTracker(save_to_file=False, log_level='error')
    tracker_gpt2.start()

    inputs = gpt2_tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = inputs["attention_mask"]

    # Generate text
    with torch.no_grad():
        output_ids = gpt2_model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )

    generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    emission_gpt2 = tracker_gpt2.stop()
    print("\n--------------------------------\n")
    print("Prompt:", prompt)
    print(f"Generated ({len(output_ids[0])} tokens):", generated_text)
    print(f"\n  > GPT-2 Prediction Emissions: {emission_gpt2:.2e} kg CO2eq")

    return generated_text, emission_gpt2