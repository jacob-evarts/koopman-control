"""JEPA-style, decoder-free latent dynamics with VICReg and MPC control.

This package is a research sibling of :mod:`koopman_control`. It keeps the same
rabbit-grass ABM and dataset layer but replaces the reconstruction-grounded,
locally-linear world model with a **pure JEPA** encoder: frames are encoded to a
latent ``z``, an action-conditioned predictor evolves ``z`` under a control
input, and collapse is prevented by VICReg alone (no decoder). Physical
macrostates are recovered post-hoc with a linear readout, and control is done
with sampling-based MPC over the learned predictor.

The predictor defaults to a linear map ``z' = A z + B u + c`` -- an empirical
choice, since the latent discovered by this objective proved near-linear in time
and a linear operator beat the nonlinear MLP on multi-step rollouts. The MLP
remains available as an ablation (``predictor="residual_mlp"``).
"""

from jepa_control.model import JEPAControl, LinearPredictor, ResidualPredictor

__all__ = ["JEPAControl", "LinearPredictor", "ResidualPredictor"]
