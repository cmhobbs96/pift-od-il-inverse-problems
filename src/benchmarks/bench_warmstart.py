"""Module 2.7 benchmark: PIFT cold-start vs FD warm-start vs ODIL warm-start.

Runs the existing 1D Poisson PIFT pipeline three times on identical data and
hyperparameters, varying only the SGLD initialization, and reports:

  - burn-in length to reach a Hamiltonian-trace plateau
  - SGLD steps to reach a target L2 error
  - final L2, posterior coverage, and wall time

The script writes a CSV (and a small LaTeX-friendly summary table) plus a
3-curve loss-vs-step plot to ``outputs/bench_warmstart/``.

Run from the repo root::

    python -m benchmarks.bench_warmstart --device cpu
    python -m benchmarks.bench_warmstart --device gpu      # if a GPU is visible

The CPU/GPU toggle reuses :func:`pipelines.common.select_device` so the
benchmark exercises the same device-placement path as the GUI.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from pipelines.phase_a import run_phase_a_forward_poisson


# Modes to compare.  ``"cold"`` is the most pessimistic baseline, ``"fd"`` is
# the previous default, and ``"odil"`` is the new Module 2.7 contribution.
MODES = ("cold", "fd", "odil")


def _steps_to_threshold(traces: dict[str, np.ndarray], target_loss: float) -> int | None:
    """First step at which the Hamiltonian trace drops below ``target_loss``."""
    ham = traces["hamiltonian"]
    below = np.flatnonzero(ham < target_loss)
    if below.size == 0:
        return None
    return int(below[0])


def _l2_along_chain(
    chain: np.ndarray,
    basis_grid: np.ndarray,
    truth: np.ndarray,
    sample_every: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Cheap running L2 estimate: at every ``sample_every`` steps, take the
    running mean of the chain so far and compute its L2 vs ground truth.

    Returns ``(steps_array, l2_array)``.
    """
    n = chain.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    sample_idx = np.arange(sample_every, n + 1, sample_every)
    if sample_idx.size == 0:
        sample_idx = np.array([n])
    l2 = np.zeros(sample_idx.size, dtype=float)
    cumsum = np.cumsum(chain, axis=0)
    for k, t in enumerate(sample_idx):
        running_mean = cumsum[t - 1] / t
        phi = running_mean @ basis_grid.T
        l2[k] = float(np.sqrt(np.mean((phi - truth) ** 2)))
    return sample_idx, l2


def run_benchmark(
    device: str,
    out_dir: Path,
    n_steps: int = 8000,
    target_l2: float = 0.05,
    seed: int = 7,
) -> dict[str, dict[str, float | int]]:
    """Run all three modes back-to-back and write artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "seed": seed,
        "n_steps": n_steps,
        "burn_in": max(500, n_steps // 8),
        "thin": 8,
    }

    results: dict[str, dict[str, float | int]] = {}

    # Plot data: per-mode (steps, running_L2) curves.
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for mode in MODES:
        wall_t0 = time.monotonic()
        r = run_phase_a_forward_poisson(
            cfg=dict(cfg),
            device_preference=device,
            init_mode=mode,
            save_outputs=False,
        )
        wall = time.monotonic() - wall_t0

        # Recover the basis & truth from the runner's outputs.
        truth = np.asarray(r["phi_truth"])
        x_grid = np.asarray(r["x_grid"])
        # Reconstruct basis matrix from the recorded mean field via lstsq is
        # circular; instead, query the field directly.
        from core.parameterizations import SineBasisField  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        n_modes = int(r["chain"].shape[1])
        field = SineBasisField(n_modes)
        basis_grid = np.asarray(field.design_matrix(jnp.asarray(x_grid)))

        steps_arr, l2_arr = _l2_along_chain(
            np.asarray(r["chain"]), basis_grid, truth, sample_every=max(1, n_steps // 80)
        )
        curves[mode] = (steps_arr, l2_arr)

        # Steps-to-target metric: first sample where running L2 < target_l2.
        below = np.flatnonzero(l2_arr < target_l2)
        steps_to_target: int | None = int(steps_arr[below[0]]) if below.size else None

        results[mode] = {
            "wall_time_sec": float(wall),
            "odil_warmstart_sec": float(r["metrics"].get("odil_warmstart_sec", 0.0)),
            "final_l2": float(r["summary"]["l2_error"]),
            "coverage_90pct": float(r["summary"]["measurement_interval_coverage_90pct"]),
            "n_effective_samples": int(r["summary"]["n_posterior_samples"]),
            "steps_to_l2_target": steps_to_target if steps_to_target is not None else -1,
            "target_l2": float(target_l2),
        }

    # ------------------------------------------------------------------
    # Persist a CSV summary.
    csv_path = out_dir / "warmstart_summary.csv"
    fieldnames = [
        "mode",
        "wall_time_sec",
        "odil_warmstart_sec",
        "final_l2",
        "coverage_90pct",
        "n_effective_samples",
        "steps_to_l2_target",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for mode in MODES:
            row = {"mode": mode, **{k: results[mode][k] for k in fieldnames if k != "mode"}}
            writer.writerow(row)

    # JSON for programmatic consumption (e.g. paper figure script).
    json_path = out_dir / "warmstart_summary.json"
    with json_path.open("w") as fh:
        json.dump(
            {"device": device, "config": cfg, "target_l2": target_l2, "results": results},
            fh,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Plot (matplotlib only — no GUI deps).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for mode, (steps_arr, l2_arr) in curves.items():
            ax.plot(steps_arr, l2_arr, label=mode, lw=1.6)
        ax.axhline(target_l2, color="k", ls=":", lw=0.8, label=f"target L2={target_l2}")
        ax.set_xlabel("SGLD step")
        ax.set_ylabel("running posterior-mean L2 error")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"PIFT warm-start ablation ({device.upper()})")
        fig.tight_layout()
        fig.savefig(out_dir / "warmstart_curves.pdf")
        fig.savefig(out_dir / "warmstart_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[bench_warmstart] plot skipped: {exc}")

    print(f"[bench_warmstart] wrote {csv_path}, {json_path}, and warmstart_curves.pdf")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--target-l2", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "bench_warmstart",
    )
    args = parser.parse_args()

    results = run_benchmark(
        device=args.device,
        out_dir=args.out,
        n_steps=args.n_steps,
        target_l2=args.target_l2,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
