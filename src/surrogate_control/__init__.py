from surrogate_control.rabbit_grass_helpers import (
    ConstantCullController,
    PIDController,
    compute_metrics,
    run_abm_replicas,
)
from surrogate_control.latent_encoding import (
    encode_grid_stack,
    grids_to_cnn_input,
    load_koopman_cnn,
    load_koopman_cnn_dynamics,
    run_abm_with_grids,
    save_trajectory_h5,
    trajectory_to_latent,
)
from surrogate_control.rabbit_grass_latent import (
    blend_latent_vector,
    fit_latent_linear_surrogate,
    fit_latent_spline_surrogates,
)
from surrogate_control.rabbit_grass_ode import fit_ode_slm, ode_rhs, simulate_ode
from surrogate_control.rabbit_grass_spline import blend_spline_state, fit_mean_trajectory_splines
from surrogate_control.surrogate_latent_model import (
    SurrogateLatentLinearModel,
    SurrogateLatentSplineModel,
    decode_latent_to_grids,
    dynamics_one_step_flat,
    latent_vector_dim,
)
from surrogate_control.surrogate_ode_model import DEFAULT_G_MAX, SurrogateODEModel
from surrogate_control.surrogate_firefly_ode_model import SurrogateFireflyODEModel
from surrogate_control.firefly_ode import (
    estimate_omega_from_traj,
    fit_firefly_ode_slm,
    initial_velocity,
    simulate_firefly_ode,
)
from surrogate_control.surrogate_spline_model import SurrogateSplineModel
from surrogate_control.firefly_metrics import compute_sync_metrics_firefly
from surrogate_control.firefly_controllers import (
    PeakSyncBeaconController,
    PeriodicBeaconController,
    make_peak_sync_controller,
    make_periodic_controller,
    sliding_peak_flashing,
    sync_control_cost,
    sync_target_count,
)
from surrogate_control.firefly_helpers import (
    ConstantFlashController,
    FireflySurrogateLatentLinearModel,
    FireflySurrogateLatentSplineModel,
    compute_metrics_firefly,
    decode_observables_firefly,
    flashing_to_cnn_input,
    run_abm_replicas_firefly,
    run_abm_with_firefly_grids,
    save_firefly_trajectory_h5,
    snapshot_firefly_grids,
    trajectory_to_latent_flashing,
)

__all__ = [
    "ConstantCullController",
    "ConstantFlashController",
    "PeakSyncBeaconController",
    "PeriodicBeaconController",
    "make_peak_sync_controller",
    "make_periodic_controller",
    "sliding_peak_flashing",
    "sync_control_cost",
    "sync_target_count",
    "FireflySurrogateLatentLinearModel",
    "FireflySurrogateLatentSplineModel",
    "compute_metrics_firefly",
    "compute_sync_metrics_firefly",
    "decode_observables_firefly",
    "flashing_to_cnn_input",
    "run_abm_replicas_firefly",
    "run_abm_with_firefly_grids",
    "save_firefly_trajectory_h5",
    "snapshot_firefly_grids",
    "trajectory_to_latent_flashing",
    "DEFAULT_G_MAX",
    "PIDController",
    "SurrogateLatentLinearModel",
    "SurrogateLatentSplineModel",
    "SurrogateODEModel",
    "SurrogateFireflyODEModel",
    "SurrogateSplineModel",
    "estimate_omega_from_traj",
    "fit_firefly_ode_slm",
    "initial_velocity",
    "simulate_firefly_ode",
    "blend_latent_vector",
    "blend_spline_state",
    "compute_metrics",
    "encode_grid_stack",
    "fit_latent_linear_surrogate",
    "fit_latent_spline_surrogates",
    "fit_mean_trajectory_splines",
    "fit_ode_slm",
    "grids_to_cnn_input",
    "load_koopman_cnn",
    "load_koopman_cnn_dynamics",
    "decode_latent_to_grids",
    "dynamics_one_step_flat",
    "latent_vector_dim",
    "ode_rhs",
    "run_abm_replicas",
    "run_abm_with_grids",
    "save_trajectory_h5",
    "simulate_ode",
    "trajectory_to_latent",
]
