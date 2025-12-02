# BER Prediction for AWGN and Rayleigh Fading Channels using ANN

A machine learning approach to predict Bit Error Rate (BER) in wireless communication systems using Artificial Neural Networks (ANN). This project simulates AWGN (Additive White Gaussian Noise) and Rayleigh fading channels, generates BER datasets, and trains ANN models to predict BER based on SNR and modulation schemes.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [ANN Architecture](#ann-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

In wireless communications, accurately predicting the Bit Error Rate (BER) is crucial for system design and optimization. Traditional methods rely on complex mathematical formulas that can be computationally expensive. This project demonstrates how Artificial Neural Networks can be trained to predict BER efficiently across different:

- **Signal-to-Noise Ratios (SNR)**: 0 to 20 dB
- **Modulation Schemes**: BPSK, QPSK, 16-QAM
- **Channel Types**: AWGN and Rayleigh Fading

## ✨ Features

- **MATLAB Dataset Generation**: Comprehensive simulation code for generating BER data with adaptive bit counting for accurate high-SNR measurements
- **Jupyter Notebook Analysis**: Complete Python implementation for data preprocessing, ANN training, and visualization
- **Multiple Modulation Schemes**: Support for BPSK, QPSK, and 16-QAM
- **Dual Channel Models**: Both AWGN and Rayleigh fading channel simulations
- **High Accuracy**: Achieves R² > 0.99 on BER prediction
- **Comprehensive Visualization**: Training curves, prediction vs. actual plots, and BER vs. SNR curves

## 📁 Project Structure

```
├── BER_PREDICTION_NOTEBOOK.ipynb    # Main Jupyter notebook with ANN implementation
├── ber_dataset_generation_matlab_code.m  # MATLAB code for dataset generation
├── ber_dataset_improved.csv         # Generated BER dataset
├── BER_PREDICTION_using_AWGN_REILEGN_CHANNEL_USING_ANN.pdf  # Project report
├── reference_paper.pdf              # Reference research paper
└── README.md                        # This file
```

## 🔬 Methodology

### Channel Models

#### AWGN Channel
The Additive White Gaussian Noise channel is the simplest model where the transmitted signal is corrupted only by white Gaussian noise:

```
y = x + n
```
where `n` is the Gaussian noise with zero mean.

#### Rayleigh Fading Channel
Models multipath propagation in environments without a direct line-of-sight path:

```
y = h·x + n
```
where `h` is the complex Rayleigh fading coefficient.

### Modulation Schemes

| Scheme | Bits/Symbol | Description |
|--------|-------------|-------------|
| BPSK   | 1           | Binary Phase Shift Keying |
| QPSK   | 2           | Quadrature Phase Shift Keying |
| 16-QAM | 4           | 16-Quadrature Amplitude Modulation |

### Adaptive Simulation Strategy

The MATLAB code implements an adaptive simulation approach:
- **High BER region (>10⁻³)**: Standard Monte Carlo simulation
- **Medium BER region (10⁻⁵ to 10⁻³)**: Extended bit counts for accuracy
- **Low BER region (<10⁻⁷)**: Theoretical estimation to avoid excessive computation

## 📊 Dataset

The dataset (`ber_dataset_improved.csv`) contains 1,230 samples with the following features:

| Column | Description |
|--------|-------------|
| `SNR_dB` | Signal-to-Noise Ratio in decibels (0-20 dB) |
| `Mod_Code` | Modulation type code (1=BPSK, 2=QPSK, 3=16QAM) |
| `Ch_Code` | Channel type code (1=AWGN, 2=Rayleigh) |
| `BER` | Bit Error Rate (target variable) |
| `Modulation` | Modulation type label |
| `Channel` | Channel type label |

## 🧠 ANN Architecture

The neural network architecture used for BER prediction:

```
Input Layer (4 features)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Output Layer (1 neuron, Linear)
```

**Key Training Parameters:**
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Target Variable**: log₁₀(BER) for better learning
- **Epochs**: 200 with early stopping
- **Validation Split**: 20%

## 🚀 Installation

### Prerequisites

- Python 3.7+
- MATLAB (for dataset generation, optional)

### Python Dependencies

```bash
pip install numpy pandas tensorflow keras scikit-learn matplotlib seaborn
```

## 💻 Usage

### 1. Generate Dataset (Optional)

If you want to generate a new dataset, run the MATLAB code:

```matlab
% Open MATLAB and run
ber_dataset_generation_matlab_code.m
```

### 2. Train and Evaluate ANN

Open and run the Jupyter notebook:

```bash
jupyter notebook BER_PREDICTION_NOTEBOOK.ipynb
```

Or use Google Colab:
1. Upload `BER_PREDICTION_NOTEBOOK.ipynb` to Google Colab
2. Upload `ber_dataset_improved.csv` to the session
3. Run all cells

## 📈 Results

### Model Performance

| Channel | R² Score | MSE (on BER) | MAE (on BER) |
|---------|----------|--------------|--------------|
| AWGN    | ~0.97    | ~0.06        | ~0.16        |
| Rayleigh| ~0.999   | ~4.23e-06    | ~1.13e-03    |

### Key Findings

- The ANN successfully learns the relationship between SNR, modulation scheme, and BER
- Rayleigh channel predictions show higher accuracy due to more consistent BER patterns
- The log transformation of BER significantly improves training stability
- Early stopping prevents overfitting while achieving optimal performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📚 References

- Reference paper included in the repository (`reference_paper.pdf`)
- Project report: `BER_PREDICTION_using_AWGN_REILEGN_CHANNEL_USING_ANN.pdf`

---

**Note**: This project is for educational and research purposes. The models and results should be validated before use in production systems.