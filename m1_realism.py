"""M1 robustness support: plant perturbations and measurement noise.

M1 scope note: these are mild, explicit non-idealities used to test whether
the M0 controller architecture tolerates an imperfect plant and imperfect
measurements. This is NOT a validated physical model of the final RM
micro-UAV.

Two mechanisms live here:

* PlantConfig / apply_plant_config: per-rollout, fixed plant changes
  (mass/inertia scaling, per-motor effectiveness, motor lag override,
  drag toggle). The controller is constructed from the nominal model and
  never learns about the perturbation -- that is what makes it uncertainty.

* NoiseConfig / MeasurementModel: a measurement layer between MuJoCo
  ground truth and the controller. Noise is freshly sampled per step from
  a seeded generator; it never touches data.qpos/qvel and never
  accumulates into the true state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from fly import ROTOR_NAMES, QuadrotorState


# Nominal asymmetric case in model actuator order (FL, FR, RR, RL):
# FL=0.97, FR=1.00, RR=0.99, RL=1.02.
M1_MOTOR_EFFECTIVENESS = (0.97, 1.00, 0.99, 1.02)
NOMINAL_MOTOR_TAU = 0.040
AIR_DENSITY = 1.2
AIR_VISCOSITY = 1.8e-5


@dataclass(frozen=True)
class PlantConfig:
    """Fixed per-rollout plant perturbations (all default to nominal).

    motor_effectiveness is in model actuator order (FL, FR, RR, RL);
    actual thrust = effectiveness * commanded thrust, implemented as the
    actuator fixed gain so it works on both the M0 direct-drive model and
    the M1 filterexact model.
    motor_tau overrides the filterexact time constant of quadrotor_m1.xml
    (it is an error on the M0 model, which has no motor dynamics).
    drag is tri-state: None keeps the XML value (quadrotor_m1.xml has
    air-like fluid params compiled in), True forces air-like density and
    viscosity on, False forces them off. Density/viscosity are runtime
    options, so drag can be toggled on either model.
    """

    mass_scale: float = 1.0
    inertia_scale: float = 1.0
    motor_effectiveness: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    motor_tau: float | None = None
    drag: bool | None = None


def apply_plant_config(
    model: mujoco.MjModel, data: mujoco.MjData, config: PlantConfig
) -> None:
    """Apply a PlantConfig to a loaded model (call after building the
    controller, so the controller keeps nominal parameters)."""

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    if body_id < 0:
        raise ValueError("model must contain body 'quadrotor'")
    model.body_mass[body_id] *= config.mass_scale
    model.body_inertia[body_id] *= config.inertia_scale

    if len(config.motor_effectiveness) != len(ROTOR_NAMES):
        raise ValueError("motor_effectiveness must have one entry per rotor")
    # Address actuators by name so ordering mistakes cannot slip in.
    for rotor_name, effectiveness in zip(ROTOR_NAMES, config.motor_effectiveness):
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, rotor_name
        )
        if actuator_id < 0:
            raise ValueError(f"missing actuator: {rotor_name}")
        if model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED:
            raise ValueError(f"{rotor_name} must use fixed gain")
        model.actuator_gainprm[actuator_id, 0] = effectiveness

    if config.motor_tau is not None:
        if config.motor_tau <= 0.0:
            raise ValueError("motor_tau must be positive")
        if not np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_FILTEREXACT):
            raise ValueError(
                "motor_tau requires filterexact actuators (quadrotor_m1.xml)"
            )
        model.actuator_dynprm[:, 0] = config.motor_tau

    if config.drag is not None:
        if config.drag:
            model.opt.density = AIR_DENSITY
            model.opt.viscosity = AIR_VISCOSITY
        else:
            model.opt.density = 0.0
            model.opt.viscosity = 0.0

    # Recompute derived constants (subtree masses, statistics) after the
    # inertial edits; the mass matrix itself is rebuilt every mj_forward.
    mujoco.mj_setConst(model, data)


@dataclass(frozen=True)
class NoiseConfig:
    """Zero-mean Gaussian measurement noise sigmas (SI units).

    attitude_sigma is the per-axis sigma of a small rotation-vector
    perturbation (radians); 0.2 deg = 0.00349 rad.
    """

    position_sigma: float = 0.010
    velocity_sigma: float = 0.020
    attitude_sigma: float = math.radians(0.2)
    angular_velocity_sigma: float = 0.010


class MeasurementModel:
    """Ground truth -> noisy measured state, with a seeded generator."""

    def __init__(self, config: NoiseConfig, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

    def measure(self, truth: QuadrotorState) -> QuadrotorState:
        config = self.config
        position = truth.position + self.rng.normal(0.0, config.position_sigma, 3)
        velocity = truth.velocity_world + self.rng.normal(
            0.0, config.velocity_sigma, 3
        )

        # Attitude: a random small rotation vector (exponential
        # coordinates) applied in the body frame, NOT independent noise on
        # quaternion components -- the result is always a valid rotation.
        delta = self.rng.normal(0.0, config.attitude_sigma, 3)
        angle = float(np.linalg.norm(delta))
        if angle > 0.0:
            axis = delta / angle
            half = 0.5 * angle
            delta_quat = np.concatenate(
                ([math.cos(half)], math.sin(half) * axis)
            )
            quaternion = np.empty(4, dtype=float)
            mujoco.mju_mulQuat(quaternion, truth.quaternion_wxyz, delta_quat)
            quaternion /= np.linalg.norm(quaternion)
        else:
            quaternion = truth.quaternion_wxyz.copy()

        rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation_flat, quaternion)
        rotation = rotation_flat.reshape(3, 3)

        angular_velocity = truth.angular_velocity_body + self.rng.normal(
            0.0, config.angular_velocity_sigma, 3
        )
        return QuadrotorState(
            position, velocity, quaternion, rotation, angular_velocity
        )
