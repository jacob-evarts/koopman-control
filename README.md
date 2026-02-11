# Neural Network Hyperparameter Optimization with PyTorch Lightning, Optuna, and Hydra

This codebase provides a template for training and hyperparameter optimization of PyTorch Lightning models using **Optuna** for HPO, **Hydra** for configuration management, and optional **Weights & Biases (W&B)** logging.

---

## Installation

The project uses **Poetry** for environment management.

1. **Clone the repository**:

```bash
git clone <repo_url>
cd <repo_dir>
```

2. **Install dependencies using Poetry:**

```bash
poetry install
```

3. **Activate virtual environment:**

```bash
poetry shell
```

---

## Configuration

This project uses **Hydra** for flexible and hierarchical configuration management. The main configuration file is located in the `conf/` directory and includes settings for:

- Model architecture (e.g., number of layers, units, activation functions)
- Training parameters (e.g., batch size, learning rate, optimizer)
- Hyperparameter search space for Optuna
- Early stopping and pruning criteria
- Logging options (including W&B setup)

You can override any configuration option via command line or by creating custom config files. For example:

```bash
python train.py model.hidden_size=128 trainer.max_epochs=50
```

---

## Running the Experiment

To start a training run or hyperparameter optimization, use the provided `train.py` script. Basic usage:

```bash
python train.py
```

This will run training with default configuration. To perform hyperparameter optimization with Optuna, specify the number of trials:

```bash
python train.py hpo.n_trials=50
```

Additional Hydra overrides can be passed to customize the experiment without modifying config files.

---

## References

- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Optuna](https://optuna.org/)
- [Hydra](https://hydra.cc/)
- [Weights & Biases](https://wandb.ai/)