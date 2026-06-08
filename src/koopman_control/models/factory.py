"""
Model factory: build Koopman model from config, trial params, and dataset properties.
"""
from omegaconf import DictConfig

from koopman_control.loaders.dataloaders import DatasetProps
from koopman_control.models.koopman_cnn_dynamics import KoopmanCNNDynamics
from koopman_control.models.koopman_cnn_wasserstein import KoopmanCNNWasserstein
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
        decode_with_pos_raw = _get_param(trial_params, cfg, "decode_with_pos")
        decode_with_pos = bool(
            getattr(cfg.model, "decode_with_pos", True)
            if decode_with_pos_raw is None
            else decode_with_pos_raw
        )
        latent_mode = str(
            _get_param(trial_params, cfg, "latent_mode")
            or getattr(cfg.model, "latent_mode", "global")
        )
        num_populations = int(
            _get_param(trial_params, cfg, "num_populations")
            or getattr(cfg.model, "num_populations", 2)
        )
        type_feature_start = int(
            _get_param(trial_params, cfg, "type_feature_start")
            or getattr(cfg.model, "type_feature_start", 7)
        )
        latent_dim_per_type_raw = _get_param(trial_params, cfg, "latent_dim_per_type")
        latent_dim_per_type = (
            int(latent_dim_per_type_raw) if latent_dim_per_type_raw is not None else None
        )
        include_control_raw = _get_param(trial_params, cfg, "include_control")
        include_control = bool(
            getattr(cfg.model, "include_control", True)
            if include_control_raw is None
            else include_control_raw
        )
        return KoopmanGNN(
            node_input_dim=dataset_props.node_input_dim,
            hidden_size=hidden_size,
            lr=lr,
            latent_dim=latent_dim,
            activation=activation,
            num_gnn_layers=num_gnn_layers,
            beta=beta,
            decode_with_pos=decode_with_pos,
            latent_mode=latent_mode,
            num_populations=num_populations,
            type_feature_start=type_feature_start,
            latent_dim_per_type=latent_dim_per_type,
            include_control=include_control,
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
        common = dict(
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
        arch = str(
            _get_param(trial_params, cfg, "arch") or getattr(cfg.model, "arch", "dynamics")
        ).lower()
        if arch in ("wasserstein", "ot", "sinkhorn"):
            return KoopmanCNNWasserstein(
                **common,
                ot_grid_size=int(_get_param(trial_params, cfg, "ot_grid_size") or 16),
                ot_epsilon=float(_get_param(trial_params, cfg, "ot_epsilon") or 0.05),
                ot_iters=int(_get_param(trial_params, cfg, "ot_iters") or 30),
                ot_mass_weight=float(_get_param(trial_params, cfg, "ot_mass_weight") or 1.0),
            )
        return KoopmanCNNDynamics(**common)
    raise ValueError(f"Unknown model_type: {dataset_props.model_type}")
