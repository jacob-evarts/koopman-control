"""
Model factory: build Koopman model from config, trial params, and dataset properties.
"""
from omegaconf import DictConfig

from koopman_control.loaders.dataloaders import DatasetProps
from koopman_control.models.koopman_cnn import KoopmanCNN
from koopman_control.models.koopman_mlp import KoopmanMLP
from koopman_control.models.koopman_gnn import KoopmanGNN


def _get_param(trial_params: dict, cfg: DictConfig, name: str):
    """Resolve a model param from trial (key may be 'model.name' or 'name') or config."""
    for key in (f"model.{name}", name):
        if key in trial_params:
            return trial_params[key]
    return getattr(cfg.model, name, None)


def build_model(
    cfg: DictConfig,
    trial_params: dict,
    dataset_props: DatasetProps):
    """
    Build a Koopman Lightning module from config, Optuna trial params, and dataset props.
    """
    hidden_size = _get_param(trial_params, cfg, "hidden_size") or cfg.model.hidden_size
    lr = _get_param(trial_params, cfg, "lr") or cfg.model.lr
    latent_dim = _get_param(trial_params, cfg, "latent_dim") or cfg.model.latent_dim
    activation = _get_param(trial_params, cfg, "activation") or cfg.model.activation

    if dataset_props.model_type == "gnn":
        return KoopmanGNN(
            node_input_dim=dataset_props.node_input_dim,
            hidden_size=hidden_size,
            lr=lr,
            latent_dim=latent_dim,
            activation=activation,
        )
    if dataset_props.model_type == "mlp":
        return KoopmanMLP(
            hidden_size=hidden_size,
            lr=lr,
            latent_dim=latent_dim,
            activation=activation,
            input_dim=dataset_props.input_dim,
        )
    if dataset_props.model_type == "cnn":
        return KoopmanCNN(
            hidden_size=hidden_size,
            lr=lr,
            latent_dim=latent_dim,
            activation=activation,
            num_channels=dataset_props.num_channels,
        )
    raise ValueError(f"Unknown model_type: {dataset_props.model_type}")
