from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from force_estimation_speed import (
    available_datasets,
    auto_optimize_forces,
    load_experiment_curvatures,
    load_experiment_force,
    load_experiment_force_vectors,
    load_experiment_shape,
    multi_force,
    sample_model_curvatures,
)

DEFAULT_DATASET = "t02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python port of estimation_MultiForce.mlx")
    parser.add_argument("--dataset", default=None, help="Experiment name, for example d01 or s08.")
    parser.add_argument("--dataset-dir", default=None, help="Directory containing the copied dataset files.")
    parser.add_argument("--n-steps", type=int, default=120, help="Discretization count used by the forward model.")
    parser.add_argument("--bias", type=float, default=8.0, help="Bias added to experimental locations.")
    parser.add_argument(
        "--initial-guess",
        type=float,
        nargs="+",
        default=None,
        help="Explicit starting vector [s1 Fx1 Fy1 s2 Fx2 Fy2 ...].",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=2.0,
        help="Keep adding one force until the loss is at or below this threshold.",
    )
    parser.add_argument(
        "--max-forces",
        type=int,
        default=6,
        help="Maximum number of force triplets to try during the automatic search.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot experimental vs. model curvatures if matplotlib is installed.",
    )
    parser.add_argument("--maxiter", type=int, default=200, help="Maximum optimizer iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else root / "dataset"
    print(f"[Stage] Start reading data from directory: {dataset_dir}")
    datasets = available_datasets(dataset_dir)
    if not datasets:
        raise FileNotFoundError(f"No datasets found in {dataset_dir}.")

    dataset = args.dataset or _default_dataset(datasets)
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset '{dataset}'. Available datasets: {', '.join(datasets)}")

    print(f"[Stage] Selected dataset: {dataset}")
    u_n_exp = load_experiment_curvatures(dataset_dir, dataset)
    print(f"[Stage] Finish reading curvature data for dataset: {dataset}")
    initial_guess = np.asarray(args.initial_guess, dtype=float) if args.initial_guess is not None else None
    if initial_guess is not None:
        print(f"[Stage] Start from user-provided initial guess with force count = {initial_guess.size // 3}")
    else:
        print("[Stage] Start automatic force-count search from 1 force.")

    try:
        print("[Stage] Start optimization search.")
        search_result = auto_optimize_forces(
            u_n_exp=u_n_exp,
            n_steps=args.n_steps,
            loss_threshold=args.loss_threshold,
            max_force_count=args.max_forces,
            initial_guess=initial_guess,
            maxiter=args.maxiter,
            verbose=True,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print("[Stage] Finish optimization search.")

    result = search_result.result
    estimated_forces = result.x
    loss = result.fun
    optimization_summary = {
        "success": result.success,
        "message": result.message,
        "nit": result.nit,
        "nfev": result.nfev,
    }

    u_n = multi_force(estimated_forces, n_steps=args.n_steps)
    u_n_model = sample_model_curvatures(u_n)
    print(f"[Stage] Start reading force data for dataset: {dataset}")
    loc_exp, force_exp = load_experiment_force(dataset_dir, dataset)
    print(f"[Stage] Finish reading force data for dataset: {dataset}")

    print("-"*20, "Optimization Summary", "-"*20)
    print(f"Loss threshold: {args.loss_threshold:.6f}")
    print("Loss by force count:")
    for attempt in search_result.history:
        print(f"  {attempt.force_count} force(s): {attempt.fun:.6f}")
    print(f"Selected force count: {result.force_count}")
    print(f"Threshold reached: {search_result.reached_threshold}")
    print(f"Loss: {loss:.6f}")
    # print(f"Estimated force vector [s, Fx, Fy, ...]: {np.array2string(estimated_forces, precision=6)}")
    print(f"Optimization success: {optimization_summary['success']}")
    print(f"Optimization message: {optimization_summary['message']}")
    print(f"Iterations: {optimization_summary['nit']}")
    print(f"Function evaluations: {optimization_summary['nfev']}")
    print("Estimated | Ground Truth")
    print(_format_estimated_ground_truth_table(estimated_forces, loc_exp + args.bias, force_exp))

    if args.plot:
        _plot_results(
            u_n_exp=u_n_exp,
            u_n_model=u_n_model,
            dataset_dir=dataset_dir,
            dataset=dataset,
            bias=args.bias,
            estimated_forces=estimated_forces,
        )


def _default_dataset(datasets: list[str]) -> str:
    if DEFAULT_DATASET in datasets:
        return DEFAULT_DATASET
    return datasets[0]


def _format_estimated_ground_truth_table(
    estimated_forces: np.ndarray,
    ground_truth_locations: np.ndarray,
    ground_truth_force_norms: np.ndarray,
) -> str:
    estimated = np.asarray(estimated_forces, dtype=float).reshape(-1, 3)
    estimated = estimated[np.argsort(estimated[:, 0])]
    estimated_locations = estimated[:, 0]
    estimated_force_norms = np.linalg.norm(estimated[:, 1:3], axis=1)

    ground_truth_locations = np.asarray(ground_truth_locations, dtype=float).reshape(-1)
    ground_truth_force_norms = np.asarray(ground_truth_force_norms, dtype=float).reshape(-1)
    ground_truth_order = np.argsort(ground_truth_locations)
    ground_truth_locations = ground_truth_locations[ground_truth_order]
    ground_truth_force_norms = ground_truth_force_norms[ground_truth_order]

    row_count = max(estimated_locations.size, ground_truth_locations.size)
    header = f"{'#':>2} | {'Estimated s':>12} | {'Ground Truth s':>14} | {'Estimated |F|':>14} | {'Ground Truth |F|':>16}"
    separator = "-" * len(header)
    rows = [header, separator]

    for index in range(row_count):
        estimated_s = _format_optional_value(estimated_locations[index] if index < estimated_locations.size else None)
        ground_truth_s = _format_optional_value(
            ground_truth_locations[index] if index < ground_truth_locations.size else None
        )
        estimated_force = _format_optional_value(
            estimated_force_norms[index] if index < estimated_force_norms.size else None
        )
        ground_truth_force = _format_optional_value(
            ground_truth_force_norms[index] if index < ground_truth_force_norms.size else None
        )
        rows.append(
            f"{index + 1:>2} | {estimated_s:>12} | {ground_truth_s:>14} | {estimated_force:>14} | {ground_truth_force:>16}"
        )

    return "\n".join(rows)


def _format_optional_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _plot_results(
    u_n_exp: np.ndarray,
    u_n_model: np.ndarray,
    dataset_dir: Path,
    dataset: str,
    bias: float,
    estimated_forces: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - exercised manually in this environment
        raise RuntimeError("matplotlib is required for --plot.") from exc

    sample_locations = np.arange(20, 281, 20)

    curvature_figure, curvature_axis = plt.subplots()
    curvature_axis.plot(sample_locations, u_n_exp.T, "o")
    curvature_axis.plot(sample_locations, u_n_model.T)
    curvature_axis.set_xlabel("Arc Length")
    curvature_axis.set_ylabel("Curvature")
    curvature_axis.set_title("Experimental vs. Model Curvatures")

    original_shape = load_experiment_shape(dataset_dir, dataset)
    ground_truth_locations, ground_truth_force_vectors = load_experiment_force_vectors(dataset_dir, dataset)
    ground_truth_locations = ground_truth_locations + bias

    estimated = np.asarray(estimated_forces, dtype=float).reshape(-1, 3)
    estimated = estimated[np.argsort(estimated[:, 0])]
    estimated_force_locations = estimated[:, 0]
    estimated_force_vectors = np.column_stack((estimated[:, 1], estimated[:, 2], np.zeros(len(estimated))))
    estimated_force_magnitudes = np.linalg.norm(estimated_force_vectors, axis=1)
    ground_truth_force_magnitudes = np.linalg.norm(ground_truth_force_vectors, axis=1)

    ground_truth_force_points = _interpolate_shape_points(original_shape, ground_truth_locations)
    estimated_force_points = _interpolate_shape_points(original_shape, estimated_force_locations)
    ground_truth_marker_sizes = _compute_marker_sizes(ground_truth_force_magnitudes)
    estimated_marker_sizes = _compute_marker_sizes(estimated_force_magnitudes)

    shape_figure = plt.figure()
    shape_axis = shape_figure.add_subplot(111, projection="3d")
    shape_axis.plot(
        original_shape[:, 0],
        original_shape[:, 1],
        original_shape[:, 2],
        color="tab:blue",
        linewidth=2.0,
    )
    # Force directions are intentionally not plotted. The force sensor and FBG
    # center frames are not calibrated, and the estimated directions are also
    # not sufficiently aligned. Marker size encodes force magnitude instead.
    if ground_truth_force_points.size:
        shape_axis.scatter(
            ground_truth_force_points[:, 0],
            ground_truth_force_points[:, 1],
            ground_truth_force_points[:, 2],
            color="tab:green",
            s=ground_truth_marker_sizes,
            alpha=0.75,
        )
    if estimated_force_points.size:
        shape_axis.scatter(
            estimated_force_points[:, 0],
            estimated_force_points[:, 1],
            estimated_force_points[:, 2],
            color="tab:red",
            s=estimated_marker_sizes,
            alpha=0.75,
        )

    shape_axis.set_xlabel("X (mm)")
    shape_axis.set_ylabel("Y (mm)")
    shape_axis.set_zlabel("Z (mm)")
    shape_axis.set_title("Original 3D Shape with Force Locations and Magnitudes")
    shape_axis.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", linewidth=2.0, label="Original Shape"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color="tab:green",
                markersize=7,
                label="Ground Truth Magnitude",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color="tab:red",
                markersize=7,
                label="Estimated Magnitude",
            ),
        ]
    )
    _set_axes_equal_3d(
        shape_axis,
        original_shape,
        ground_truth_force_points,
        estimated_force_points,
    )
    plt.show()


def _interpolate_shape_points(shape: np.ndarray, positions: np.ndarray) -> np.ndarray:
    shape = np.asarray(shape, dtype=float)
    positions = np.asarray(positions, dtype=float).reshape(-1)
    if shape.size == 0 or positions.size == 0:
        return np.zeros((0, 3), dtype=float)

    index_axis = np.linspace(1.0, float(shape.shape[0]), shape.shape[0])
    clipped_positions = np.clip(positions, index_axis[0], index_axis[-1])
    return np.column_stack(
        [
            np.interp(clipped_positions, index_axis, shape[:, 0]),
            np.interp(clipped_positions, index_axis, shape[:, 1]),
            np.interp(clipped_positions, index_axis, shape[:, 2]),
        ]
    )


def _compute_marker_sizes(
    magnitudes: np.ndarray,
    min_size: float = 30.0,
    max_size: float = 100.0,
) -> np.ndarray:
    magnitudes = np.asarray(magnitudes, dtype=float).reshape(-1)
    if magnitudes.size == 0:
        return np.zeros(0, dtype=float)

    max_magnitude = float(np.max(magnitudes))
    min_magnitude = float(np.min(magnitudes))
    if max_magnitude <= 0.0 or np.isclose(max_magnitude, min_magnitude):
        return np.full(magnitudes.shape, (min_size + max_size) / 2.0, dtype=float)

    normalized = (magnitudes - min_magnitude) / (max_magnitude - min_magnitude)
    return min_size + normalized * (max_size - min_size)


def _set_axes_equal_3d(ax, *point_sets: np.ndarray) -> None:
    valid_sets = [np.asarray(points, dtype=float).reshape(-1, 3) for points in point_sets if np.asarray(points).size]
    if not valid_sets:
        return

    all_points = np.vstack(valid_sets)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


if __name__ == "__main__":
    main()
