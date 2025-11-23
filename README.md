# Setup

To get started, create and activate a virtual environment, then install the required dependencies:

```bash
pip3 install uv
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
