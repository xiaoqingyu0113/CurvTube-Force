from __future__ import annotations

import numpy as np

ROD_LENGTH_M = 0.290
ROD_LENGTH_MM = 290.0
SAMPLE_LOCATIONS_MM = np.arange(20.0, 281.0, 20.0)


def get_stiffness() -> np.ndarray:
    od = 1.397e-3
    inner_diameter = 1.118e-3
    elastic_modulus = 67e9 # calibrated by three of the test
    shear_modulus = elastic_modulus / (2 * (1 + 0.3))

    ix = np.pi / 4 * (od / 2) ** 4 - np.pi / 4 * (inner_diameter / 2) ** 4
    iy = ix
    iz = ix + iy

    return np.array(
        [
            [elastic_modulus * ix, 0.0, 0.0],
            [0.0, elastic_modulus * iy, 0.0],
            [0.0, 0.0, shear_modulus * iz],
        ],
        dtype=float,
    )


def multi_force(force_vector: np.ndarray, n_steps: int = 290) -> np.ndarray:
    forces = np.asarray(force_vector, dtype=float).reshape(-1)
    if forces.size % 3 != 0:
        raise ValueError("Force vector must contain triplets of [s, Fx, Fy].")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")

    ds = ROD_LENGTH_M / n_steps
    nodal_forces = np.zeros((3, n_steps), dtype=float)
    locations = np.linspace(1.0, ROD_LENGTH_MM, n_steps)

    for s, fx, fy in forces.reshape(-1, 3):
        contribution = np.array([fx, fy, 0.0], dtype=float)

        if s <= locations[0]:
            nodal_forces[:, 0] += contribution
            continue
        if s >= locations[-1]:
            nodal_forces[:, -1] += contribution
            continue

        right_index = int(np.searchsorted(locations, s, side="left"))
        left_index = right_index - 1
        left_location = locations[left_index]
        right_location = locations[right_index]
        width = right_location - left_location

        nodal_forces[:, left_index] += contribution * (right_location - s) / width
        nodal_forces[:, right_index] += contribution * (s - left_location) / width

    un_history = _get_un_n(ds, n_steps, nodal_forces)
    return un_history[:2, :]


def sample_model_curvatures(u_n: np.ndarray) -> np.ndarray:
    n_steps = u_n.shape[1]
    s = np.linspace(0.0, ROD_LENGTH_MM, n_steps)
    return np.vstack(
        (
            np.interp(SAMPLE_LOCATIONS_MM, s, u_n[0, :]),
            np.interp(SAMPLE_LOCATIONS_MM, s, u_n[1, :]),
        )
    )


def estimation_loss(force_vector: np.ndarray, u_n_exp: np.ndarray, n_steps: int) -> float:
    u_n = multi_force(force_vector, n_steps=n_steps)
    u_n_model = sample_model_curvatures(u_n)
    residual = u_n_model - u_n_exp
    return float(np.sum(residual**2))


def _get_un_n(ds: float, n_steps: int, nodal_forces: np.ndarray) -> np.ndarray:
    reversed_forces = np.flip(nodal_forces, axis=1)
    un0 = np.zeros(5, dtype=float)
    un_history = np.flip(_int_to_distal_n(un0, ds, n_steps, reversed_forces, _ode_un), axis=1)
    un_history[:2, :] *= -1.0
    return un_history


def _int_to_distal_n(
    x0: np.ndarray,
    ds: float,
    n_steps: int,
    nodal_forces: np.ndarray,
    fun,
) -> np.ndarray:
    stiffness = np.diag(get_stiffness())
    params = np.concatenate((stiffness, nodal_forces[:, 0]), dtype=float)

    state = np.asarray(x0, dtype=float).copy()
    history = np.zeros((state.size, n_steps), dtype=float)
    history[:, 0] = state

    for i in range(n_steps - 1):
        params[3:6] = nodal_forces[:, i] / ds
        state = _rk4_step(state, ds, lambda s, x: fun(s, x, params))
        history[:, i + 1] = state

    return history


def _rk4_step(state: np.ndarray, dt: float, fun) -> np.ndarray:
    c = np.array([0.0, 0.5, 0.5, 1.0])
    b = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    a = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    k = np.zeros((state.size, 4), dtype=float)
    for i in range(4):
        k[:, i] = fun(c[i] * dt, state + dt * (k @ a[i, :]))
    return state + dt * (k @ b)


def _ode_un(_s: float, state: np.ndarray, params: np.ndarray) -> np.ndarray:
    k1, k2 = params[0], params[1]
    fx, fy = params[3], params[4]

    ux, uy, nx, ny, nz = state
    xdot = np.zeros_like(state)
    xdot[0] = -ny / k1
    xdot[1] = nx / k2
    xdot[2] = -fx + uy * nz
    xdot[3] = -fy - ux * nz
    xdot[4] = -uy * nz + ux * ny
    return xdot
