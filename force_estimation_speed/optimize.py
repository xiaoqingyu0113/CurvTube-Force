from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import estimation_loss

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - exercised manually in this environment
    minimize = None


@dataclass
class OptimizationResult:
    x: np.ndarray
    fun: float
    success: bool | None
    message: str
    nit: int | None
    nfev: int | None

    @property
    def force_count(self) -> int:
        return int(self.x.size // 3)


@dataclass
class ForceCountSearchResult:
    result: OptimizationResult
    history: list[OptimizationResult]
    loss_threshold: float
    reached_threshold: bool


def build_initial_guess(force_count: int = 2) -> np.ndarray:
    if force_count < 1:
        raise ValueError("force_count must be at least 1.")

    positions = 290.0 * np.arange(1, force_count + 1, dtype=float) / (force_count + 1)
    guess = np.zeros(force_count * 3, dtype=float)
    guess[0::3] = positions
    return guess


def optimize_forces(
    u_n_exp: np.ndarray,
    n_steps: int,
    initial_guess: np.ndarray | None = None,
    maxiter: int = 200,
) -> OptimizationResult:
    if minimize is None:
        raise RuntimeError("SciPy is required for optimization. Install the packages in requirements.txt.")

    x0 = np.asarray(initial_guess, dtype=float) if initial_guess is not None else build_initial_guess()
    if x0.size % 3 != 0:
        raise ValueError("Initial guess must contain triplets of [s, Fx, Fy].")

    force_count = x0.size // 3
    lower = np.tile(np.array([1.0, -1.6, -1.6], dtype=float), force_count)
    upper = np.tile(np.array([285.0, 1.6, 1.6], dtype=float), force_count)
    bounds = list(zip(lower, upper))

    result = minimize(
        lambda x: estimation_loss(x, u_n_exp=u_n_exp, n_steps=n_steps),
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "gtol": 1e-2, "maxls": 50},
    )

    return OptimizationResult(
        x=np.asarray(result.x, dtype=float),
        fun=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        nit=getattr(result, "nit", None),
        nfev=getattr(result, "nfev", None),
    )

def auto_optimize_forces(
    u_n_exp: np.ndarray,
    n_steps: int,
    loss_threshold: float = 2.0,
    max_force_count: int = 6,
    initial_guess: np.ndarray | None = None,
    maxiter: int = 200,
    verbose: bool = False,
) -> ForceCountSearchResult:
    if max_force_count < 1:
        raise ValueError("max_force_count must be at least 1.")
    if loss_threshold < 0:
        raise ValueError("loss_threshold must be non-negative.")

    if initial_guess is None:
        current_guess = build_initial_guess(1)
    else:
        current_guess = np.asarray(initial_guess, dtype=float).reshape(-1)
        if current_guess.size % 3 != 0:
            raise ValueError("Initial guess must contain triplets of [s, Fx, Fy].")
        if current_guess.size // 3 > max_force_count:
            raise ValueError("Initial guess uses more forces than max_force_count allows.")

    history: list[OptimizationResult] = []
    while True:
        force_count = int(current_guess.size // 3)
        if verbose:
            print(f"[Stage] Start with force count = {force_count}")
            print(f"[Stage] Start optimization for force count = {force_count}")
        result = optimize_forces(
            u_n_exp=u_n_exp,
            n_steps=n_steps,
            initial_guess=current_guess,
            maxiter=maxiter,
        )
        if verbose:
            print(
                f"[Stage] Finish optimization for force count = {force_count}. "
                f"Loss = {result.fun:.6f}, success = {result.success}"
            )
        history.append(result)

        if result.fun <= loss_threshold:
            if verbose:
                print(
                    f"[Stage] Loss threshold reached at force count = {force_count}. "
                    f"Threshold = {loss_threshold:.6f}"
                )
            return ForceCountSearchResult(
                result=result,
                history=history,
                loss_threshold=loss_threshold,
                reached_threshold=True,
            )

        if result.force_count >= max_force_count:
            if verbose:
                print(
                    f"[Stage] Reached max force count = {max_force_count} "
                    f"without meeting the loss threshold."
                )
            return ForceCountSearchResult(
                result=result,
                history=history,
                loss_threshold=loss_threshold,
                reached_threshold=False,
            )

        if verbose:
            print(f"[Stage] Add one more force and continue the search.")
        current_guess = expand_force_guess(result.x)


def expand_force_guess(force_vector: np.ndarray) -> np.ndarray:
    forces = np.asarray(force_vector, dtype=float).reshape(-1, 3)
    next_position = _next_force_position(forces[:, 0])
    expanded = np.vstack((forces, np.array([next_position, 0.0, 0.0], dtype=float)))
    expanded = expanded[np.argsort(expanded[:, 0])]
    return expanded.reshape(-1)


def _next_force_position(positions: np.ndarray) -> float:
    if positions.size == 0:
        return float(build_initial_guess(1)[0])

    sorted_positions = np.sort(np.asarray(positions, dtype=float))
    bounded_positions = np.concatenate(([1.0], sorted_positions, [285.0]))
    gaps = np.diff(bounded_positions)
    gap_index = int(np.argmax(gaps))
    return float((bounded_positions[gap_index] + bounded_positions[gap_index + 1]) / 2.0)
