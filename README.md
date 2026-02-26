# TrackSelectorDNN

Machine Learning framework for training pixel track selection models for the Phase-2 High Level Trigger (HLT).
This repository provides tools to preprocess detector data, construct datasets, configure and train deep learning models (including dense and graph-based architectures), and evaluate performance for track selection tasks.

The repository includes:

- Data preprocessing pipelines
- Modular dataset and model factories
- Configurable training and hyperparameter tuning workflows
- Support for multiple network architectures
- Utilities for visualization and performance monitoring

The configuration system is YAML-driven, supporting different model architectues and datasets.

## Repository Structure
```
TrackSelectorDNN/
│
├── TrackSelectorDNN/
│   ├── configs/          # Training and model configuration files
│   ├── data_manager/     # Dataset construction and loading utilities
│   ├── models/           # Network architectures and model registry
│   ├── train/            # Optimizer and training utilities
│   ├── tune/             # Hyperparameter tuning workflows
│   └── utils/            # Configuration and helper utilities
│
├── trackkit/             # Additional preprocessing and visualization tools
│
├── notebooks/
│   ├── ReadRootData.ipynb
│   ├── PrepareOneHot.ipynb
│   ├── PrepareUnnormalizedData.ipynb
│   ├── GenerateSyntheticData.ipynb
│   └── InvestigateFakes.ipynb
│
├── requirements.txt      # Python dependencies
├── setup.py              # Package installation
└── README.md
```

## Installation

1. Clone Repository
```
git clone https://github.com/EmanueleCoradin/TrackSelectorDNN
cd TrackSelectorDNN
```
2. Create Environment (Recommended)
```
conda create -n trackselector python=3.10
conda activate trackselector
```
3. Install Dependencies
```
pip install -r requirements.txt
pip install -e .
```

## Configuration

All training parameters are controlled via YAML configuration files located in `TrackSelectorDNN/configs/`

Examples:

- `base.yaml` – Standard DNN training configuration
- `base_GNN.yaml` – Graph Neural Network setup
- `preselector.yaml` – Pre-selection model configuration

Configurations typically define:

- Model architecture
- Input feature schema
- Dataset parameters
- Optimizer and scheduler
- Training hyperparameters
- Logging options

## Training and Hyper Parameter tuning

Training and Hyperparameter Tuning are performed via configurable training scripts. 
A training pipeline is implemented with RayTune (see https://github.com/EmanueleCoradin/TrackSelectorDNN-Ray)

