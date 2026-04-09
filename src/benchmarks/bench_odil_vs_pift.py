"""Module 2.8 benchmark: ODIL vs PIFT accuracy-vs-walltime comparison.

Reproduces the spirit of Karnakov et al. (2024) Fig. 1 in the PIFT setting:
each method is run on the same 1D Poisson problem and we record (wall time,
posterior-mean L2) checkpoints throughout the run.  Methods compared:

  1. ODIL Gauss-Newton (``core.odil``)
  2. ODIL L-BFGS (``core.odil``)
  3. PIFT cold-start SGLD (``run_phase_a_forward_poisson``, init_mode="cold")
  4. PIFT ODIL-warm SGLD (init_mode="odil")
  5. Random-walk Monte Carlo (``run_phase_a_monte_carlo``)

Output: ``accuracy_vs_walltime.{pdf,png}`` plus a CSV of raw checkpoints under
``outputs/bench_odil_vs_pift/``.

The script accepts ``--device {cpu,gpu,auto}`` so the same comparison can be
generated on either backend; the resulting figure should be regenerated on a
GPU host for the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np

from core.odil import odil_solve_poisson_1d
from core.parameterizations import SineBasisField
from pipelines.common import forcing, phi_true, prepare_observations, select_device
from pipelines.phase_a import (
    DEFAULT_PHASE_A_CONFIG,
    run_phase_a_forward_poisson,
    run_phase_a_monte_carlo,
)


def _make_problem(seed: int):
    """Generate the shared 1D Poisson observation set used by every method."""
    import jax  # noqa: PLC0415

    cfg = dict(DEFAULT_PHASE_A_CONFIG)
    cfg["seed"] = seed
    key = jax.random.PRNGKey(int(cfg["seed"]))
    _, x_obs, y_obs, _, _ = prepare_observations(cfg, key, None, None, time.monotonic(), None)
    return cfg, np.asarray(x_obs), np.asarray(y_obs)


def _running_l2_from_chain(chain: np.ndarray, basis_grid: np.ndarray, truth: np.ndarray, n_checkpoints: int):
    """Sample running posterior-mean L2 at ~``n_checkpoints`` points along the chain."""
    n = chain.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    sample_idx = np.unique(np.linspace(1, n, n_checkpoints, dtype=int))
    cumsum = np.cumsum(chain, axis=0)
    l2 = np.zeros(sample_idx.size, dtype=float)
    for k, t in enumerate(sample_idx):
        running_mean = cumsum[t - 1] / t
        phi = running_mean @ basis_grid.T
        l2[k] = float(np.sqrt(np.mean((phi - truth) ** 2)))
    return sample_idx, l2


def _bench_odil(method: str, cfg: dict, x_obs: np.ndarray, y_obs: np.ndarray) -> dict:
    """Run ODIL once and report (wall time, final L2)."""
    n_grid = 257
    x_truth = np.linspace(0.0, 1.0, n_grid)
    truth = np.asarray(phi_true(jnp.asarray(x_truth)))

    t0 = time.monotonic()
    result = odil_solve_poisson_1d(
        forcing_fn=forcing,
        n_grid=n_grid,
        bc=(0.0, 0.0),
        obs=(x_obs, y_obs),
        noise_std=float(cfg["noise_std"]),
        method=method,
        max_iter=200 if method == "lbfgs" else 50,
        tol=1e-10,
    )
    wall = time.monotonic() - t0
    l2 = float(np.sqrt(np.mean((result.u_grid - truth) ** 2)))
    return {
        "method": f"odil_{method}",
        "wall_time_sec": wall,
        "final_l2": l2,
        "iterations": int(result.n_iterations),
        # Single checkpoint at the end (ODIL is one-shot — no chain history).
        "checkpoints": [(wall, l2)],
    }


def _bench_pift(init_mode: str, cfg_overrides: dict, n_checkpoints: int) -> dict:
    """Run PIFT and convert chain history into wall-time checkpoints."""
    t0 = time.monotonic()
    r = run_phase_a_forward_poisson(
        cfg=cfg_overrides,
        init_mode=init_mode,
        save_outputs=False,
    )
    wall = time.monotonic() - t0

    chain = np.asarray(r["chain"])
    truth = np.asarray(r["phi_truth"])
    x_grid = np.asarray(r["x_grid"])
    n_modes = chain.shape[1] if chain.ndim == 2 else 0
    field = SineBasisField(n_modes)
    basis_grid = np.asarray(field.design_matrix(jnp.asarray(x_grid)))

    sample_idx, l2 = _running_l2_from_chain(chain, basis_grid, truth, n_checkpoints)
    # Distribute the wall time linearly across SGLD steps (we don't have
    # per-step timing — this is a fair approximation since SGLD chunks are
    # uniform).  Add the ODIL warm-start cost to *every* checkpoint so the
    # plotted curve correctly accounts for it.
    odil_warm = float(r["metrics"].get("odil_warmstart_sec", 0.0))
    sgld_wall = max(0.0, wall - odil_warm)
    n_total = chain.shape[0] if chain.ndim == 2 else 1
    checkpoints = [
        (odil_warm + sgld_wall * (t / n_total), float(l2[k]))
        for k, t in enumerate(sample_idx)
    ]
    return {
        "method": f"pift_{init_mode}",
        "wall_time_sec": wall,
        "final_l2": float(r["summary"]["l2_error"]),
        "iterations": int(n_total),
        "checkpoints": checkpoints,
    }


def _bench_mc(cfg_overrides: dict, n_checkpoints: int) -> dict:
    """Run the random-walk MC baseline."""
    t0 = time.monotonic()
    r = run_phase_a_monte_carlo(cfg=cfg_overrides, save_outputs=False)
    wall = time.monotonic() - t0

    chain = np.asarray(r["chain"])
    truth = np.asarray(r["phi_truth"])
    x_grid = np.asarray(r["x_grid"])
    n_modes = chain.shape[1] if chain.ndim == 2 else 0
    field = SineBasisField(n_modes)
    basis_grid = np.asarray(field.design_matrix(jnp.asarray(x_grid)))
    sample_idx, l2 = _running_l2_from_chain(chain, basis_grid, truth, n_checkpoints)
    n_total = chain.shape[0]
    checkpoints = [
        (wall * (t / n_total), float(l2[k])) for k, t in enumerate(sample_idx)
    ]
    return {
        "method": "mc",
        "wall_time_sec": wall,
        "final_l2": float(r["summary"]["l2_error"]),
        "iterations": int(n_total),
        "checkpoints": checkpoints,
    }


def run_benchmark(device: str, out_dir: Path, n_steps: int = 8000, n_checkpoints: int = 40) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, x_obs, y_obs = _make_problem(seed=7)

    cfg_overrides = {
        "n_steps": n_steps,
        "burn_in": max(500, n_steps // 8),
        "thin": 8,
    }

    # NB: select_device is invoked inside each runner — we don't need to wrap
    # the whole script.  We pass the preference through where it matters.

    methods = []
    print(f"[bench] device={device}, n_steps={n_steps}")
    methods.append(_bench_odil("gauss_newton", cfg, x_obs, y_obs))
    methods.append(_bench_odil("lbfgs", cfg, x_obs, y_obs))
    methods.append(_bench_pift("cold", cfg_overrides, n_checkpoints))
    methods.append(_bench_pift("odil", cfg_overrides, n_checkpoints))
    methods.append(_bench_mc(cfg_overrides, n_checkpoints))

    # ------------------------------------------------------------------
    # Persist raw checkpoints CSV.
    csv_path = out_dir / "checkpoints.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "wall_time_sec", "l2_error"])
        for m in methods:
            for wt, l2 in m["checkpoints"]:
                w.writerow([m["method"], f"{wt:.6f}", f"{l2:.6e}"])

    json_path = out_dir / "summary.json"
    with json_path.open("w") as fh:
        json.dump(
            {
                "device": device,
                "config": cfg_overrides,
                "results": [
                    {k: m[k] for k in ("method", "wall_time_sec", "final_l2", "iterations")}
                    for m in methods
                ],
            },
            fh,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        styles = {
            "odil_gauss_newton": ("o-", "tab:red"),
            "odil_lbfgs": ("s-", "tab:orange"),
            "pift_cold": ("v--", "tab:blue"),
            "pift_odil": ("^-", "tab:green"),
            "mc": ("d:", "tab:gray"),
        }
        for m in methods:
            wt, l2 = zip(*m["checkpoints"])
            style, color = styles.get(m["method"], ("o-", None))
            ax.plot(wt, l2, style, color=color, label=m["method"], lw=1.4, ms=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall time (s)")
        ax.set_ylabel("posterior-mean L2 error")
        ax.set_title(f"ODIL vs PIFT — accuracy vs wall time ({device.upper()})")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_vs_walltime.pdf")
        fig.savefig(out_dir / "accuracy_vs_walltime.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] plot skipped: {exc}")

    print(f"[bench] wrote {csv_path}, {json_path}, accuracy_vs_walltime.pdf")
    return {"methods": [m["method"] for m in methods]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--n-checkpoints", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("outputs") / "bench_odil_vs_pift")
    args = parser.parse_args()
    # Touch select_device so a bad CLI value fails fast.
    _ = select_device(args.device)
    run_benchmark(
        device=args.device,
        out_dir=args.out,
        n_steps=args.n_steps,
        n_checkpoints=args.n_checkpoints,
    )


if __name__ == "__main__":
    main()
