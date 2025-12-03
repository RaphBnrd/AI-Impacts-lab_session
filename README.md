# Description

This project provides a hands-on environment to explore the behavior and environmental impact of various "AI" models. Using the [CodeCarbon library](https://codecarbon.io/), it automatically tracks energy consumption and CO₂ emissions during model execution.

**Key Features:**
- Image Classification: Compare simple MLPs and CNNs on the MNIST dataset.
- Image Generation: Experiment with CVAEs, PCA-based models, and probabilistic pixel approaches.
- Text Generation: Test BERT (fill-in-the-blank) and GPT-2 (free text) models.

**How It’s Structured:**
- A Streamlit app for interactive exploration.
- Two Jupyter notebooks—one introductory, one in-depth—for deeper analysis.

**Why It Matters:**
- Understand how different machine learning models perform in practice.
- Assess the environmental impact of model choices.
- Explore ways to mitigate environmental impact of machine learning depending on the task.

# Overview of the content

This project includes a variety of visual outputs to help you explore how different models behave — and how much CO₂ they emit.

For the image-related tasks, we use the classic MNIST dataset of handwritten digits:

<!-- ![MNIST Dataset](images/01-MNIST_dataset.png) -->
<!-- (1259 × 302) -> height = width * 302/1259 -->
<img src="images/01-MNIST_dataset.png" alt="MNIST Dataset" width="60%" >

## 🏷️ Image Classification

For classification, you can compare MLP and CNN models and track useful metrics during training — including how CO₂ emissions scale with the number of model parameters:
<!-- (1457 × 804) -> height = width * 804/1457 -->
<img src="images/02-Image_Classification-CO2_params.png" alt="Image Classification CO2 Emissions" width="80%">

## 🎨 Image Generation

The image generation section explores several approaches such as CVAEs, PCA-based models, and probabilistic pixel models.
Each experiment includes the CO₂ emissions for both training and inference.

Below is an example using a CVAE:

<table>
<tr>
<td valign="top" width="20%">
  <strong>C-VAE Generated Images</strong><br>
  <!-- (81 × 844) -> height = width * 844/81 -->
  <img src="images/03-Image_Generation-example_generations.png" alt="C-VAE Generated Images" height="400" width="38">
</td>
<td valign="top" width="80%">
  <strong>C-VAE CO₂ Emissions (Training)</strong><br>
    <!-- (1067 × 222) -> height = width * 222/1067 -->
  <img src="images/03-Image_Generation-train_emissions.png" alt="C-VAE Train Emissions" width="90%"><br><br>

  <strong>C-VAE CO₂ Emissions (Generation)</strong><br>
  <!-- (1067 × 222) -> height = width * 222/1067 -->
  <img src="images/03-Image_Generation-generation_emissions.png" alt="C-VAE Generation Emissions" width="90%">
</td>
</tr>
</table>


## 💬 Text Generation

Finally, the text generation examples showcase emissions from inference using models like BERT (masked prediction) and GPT-2 (autoregressive generation). 

Bert example on the sentence: `The capital of France is \[MASK\].`
<!-- (1523 × 727) -> height = width * 727/1523 -->
<img src="images/04-Text_Generation-bert.png" alt="BERT Example" width="50%" >

GPT-2 example on the prompt: `The future of AI is`
<!-- (1012 × 912) -> height = width * 912/1012 -->
<img src="images/04-Text_Generation-gpt2.png" alt="GPT-2 Example" width="50%" >

# Setup

If you don't have `uv` installed (recommended for faster virtual environment creation and management), you can do so via pip:

```bash
pip3 install uv
```

To get started, create and activate a virtual environment, then install the required dependencies:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

*NB: To exit the virtual environment later, simply run: `deactivate`*

# Usage

You can explore the project using one of the following three options.

Each option provides access to similar content but with different levels of interactivity and detail:

1. **Streamlit App** (Interactive Interface): 
Launch the interactive dashboard with `streamlit run streamlit_app/Home.py`. 
This is the most user-friendly way to navigate the experiments and visualize results.
2. **Jupyter Notebook — Simplified Version** (`main_simpler.ipynb`): 
This notebook highlights the results with minimal code, ideal for quick exploration.
3. **Jupyter Notebook — Full Version** (`main.ipynb`): 
This version exposes more of the underlying implementations and is suitable for studying or modifying the code.

# Environmental Impact Tracking

The environmental impact tracking is done using the `codecarbon` library. You can find more information about it on the [website](https://codecarbon.io/) or in the [documentation](https://mlco2.github.io/codecarbon/index.html).
