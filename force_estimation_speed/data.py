from __future__ import annotations

from pathlib import Path

import numpy as np


def available_datasets(exp_dir: str | Path) -> list[str]:
    exp_path = Path(exp_dir)
    names = []
    for path in sorted(exp_path.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if (path / name).exists() and (path / f"{name}_straight").exists():
            names.append(name)
    return names


def load_experiment_curvatures(exp_dir: str | Path, dataset: str) -> np.ndarray:
    case_path = _case_path(exp_dir, dataset)
    bend_table = np.loadtxt(case_path / dataset)
    straight_table = np.loadtxt(case_path / f"{dataset}_straight")

    u_straight = _curvature_to_u(straight_table[:, 0], straight_table[:, 1])
    u_bend = _curvature_to_u(bend_table[:, 0], bend_table[:, 1])

    curvatures = (u_bend - u_straight) * 100.0
    sampled_columns = np.concatenate(([0], np.arange(19, min(curvatures.shape[1], 740), 20)))
    if sampled_columns.size < 14:
        raise ValueError(
            f"Dataset '{dataset}' does not contain enough samples for the MATLAB indexing scheme."
        )
    return curvatures[:, sampled_columns[-14:]]


def load_experiment_force(exp_dir: str | Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    loc, force = load_experiment_force_vectors(exp_dir, dataset)
    force_norm = np.linalg.norm(force, axis=1)
    return np.atleast_1d(loc).astype(float), force_norm.astype(float)


def load_experiment_force_vectors(exp_dir: str | Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    case_path = _case_path(exp_dir, dataset)
    force = np.loadtxt(case_path / f"{dataset}_f.txt", ndmin=2)
    loc = np.loadtxt(case_path / f"{dataset}_loc.txt", ndmin=1)
    return np.atleast_1d(loc).astype(float), np.asarray(force, dtype=float)


def load_experiment_shape(exp_dir: str | Path, dataset: str) -> np.ndarray:
    case_path = _case_path(exp_dir, dataset)
    bend_table = np.loadtxt(case_path / dataset)
    shape = bend_table[-290:, 2:5] * 10.0
    return np.asarray(shape, dtype=float)


def _case_path(exp_dir: str | Path, dataset: str) -> Path:
    case_path = Path(exp_dir) / dataset
    if not case_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {case_path}")
    return case_path


def _curvature_to_u(curvature: np.ndarray, angle: np.ndarray) -> np.ndarray:
    ux = -curvature * np.sin(angle)
    uy = curvature * np.cos(angle)
    return np.vstack((ux, uy))
