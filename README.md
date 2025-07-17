# Few-Shot Analytical

This repository contains code for running few-shot learning experiments with different analytical models including binary classification, linear regression, and Hidden Markov Models (HMM).

## Features

- **Binary Classification**: Prototype learning with class-dependent noise
- **Linear Regression**: Linear regression with concatenated noise encoding
- **HMM Models**: Flat and chain latent variable models with binomial MLE

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd fewshot_analytical
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main experiment script:
```bash
python fewshot_analytical.py
```

This will:
- Execute all three experiment configurations (Binary Classification, Linear Regression, HMM)
- Generate combined plots with subplots for each experiment
- Save results as PNG and PDF files in a `plots/` directory

## Output

The script generates:
- `plots/fewshot_analytical_combined_subplots.png` - Combined visualization of all experiments
- `plots/fewshot_analytical_combined_subplots.pdf` - Vector format of the combined plot

## Dependencies

- numpy (≥1.21.0)
- matplotlib (≥3.5.0)
- sympy (≥1.10.0)
- scipy (≥1.7.0)
- compose (≥1.0.0)

## Requirements

- Python 3.7 or higher
- pip package manager 