"""
Model factory: build Koopman model from config, trial params, and dataset properties.
"""
from omegaconf import DictConfig

from koopman_control.loaders.dataloaders import DatasetProps
from koopman_control.models.koopman_cnn_dynamics import KoopmanCNNDynamics
from koopman_control.models.koopman_mlp import KoopmanMLP
from koopman_control.models.koopman_gnn import KoopmanGNN

_BETA_KOOP = 5.0
_BETA_PRED = 5.0
_BETA_RECON = 0.2
_SPATIAL_LATENT_CHANNELS = 16
_INPUT_SIZE = 64


def _get_param(trial_params: dict, cfg: DictConfig, name: str):
    """Resolve a model param from trial (key may be 'model.name' or 'name') or config."""
    for key in (f"model.{name}", name):
        if key in trial_params:
            return trial_params[key]
    return getattr(cfg.model, name, None)


def build_model(
    cfg: DictConfig,
    trial_params: dict,
    dataset_props: DatasetProps,
):
    """
    Build a Koopman Lightning module from config, Optuna trial params, and dataset props.
    """
    hidden_size = _get_param(trial_params, cfg, "hidden_size") or cfg.model.hidden_size
    lr = _get_param(trial_params, cfg, "lr") or cfg.model.lr
    latent_dim = _get_param(trial_params, cfg, "latent_dim") or cfg.model.latent_dim
    activation = _get_param(trial_params, cfg, "activation") or cfg.model.activation

    if dataset_props.model_type == "gnn":
        num_gnn_layers = int(
            _get_param(trial_params, cfg, "num_gnn_layers")
            or getattr(cfg.model, "num_gnn_layers", 1)
        )
        beta = float(_get_param(trial_params, cfg, "beta") or getattr(cfg.model, "beta", 1.0))
        return KoopmanGNN(
            node_input_dim=dataset_props.node_input_dim,
            hidden_size=hidden_size,
            lr=lr,
            latent_dim=latent_dim,
            activation=activation,
            num_gnn_layers=num_gnn_layers,
            beta=beta,
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
        rollout_horizon = int(
            _get_param(trial_params, cfg, "rollout_horizon")
            or getattr(cfg.model, "rollout_horizon", 1)
        )
        return KoopmanCNNDynamics(
            hidden_size=hidden_size,
            lr=lr,
            latent_dim=latent_dim,
            activation=activation,
            num_channels=dataset_props.num_channels,
            input_size=_INPUT_SIZE,
            spatial_latent_channels=_SPATIAL_LATENT_CHANNELS,
            beta_koop=_BETA_KOOP,
            beta_pred=_BETA_PRED,
            beta_recon=_BETA_RECON,
            rollout_horizon=rollout_horizon,
        )
    raise ValueError(f"Unknown model_type: {dataset_props.model_type}")
