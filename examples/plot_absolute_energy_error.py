"""Export the online absolute energy error logged by a one-geometry training run."""

import argparse
import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHEMICAL_ACCURACY_MHA = 1000 / 627.5


def read_metrics(source):
    with h5py.File(source, "r", swmr=True) as stream:
        group = stream["metrics"]
        columns = [group[key] for key in ("mol_idx", "E_loc/mean/ewm", "E_ewm_vs_ref/mae")]
        count = min(len(column) for column in columns)
        indices, energy, error = (column[:count] for column in columns)
    if not count:
        raise KeyError("No energy metrics have been written yet.")
    if any(array.shape != (count, 1) for array in (indices, energy, error)) or np.any(indices != 0):
        raise ValueError("Expected one geometry with ID 0 throughout the training run.")
    if not np.isfinite(energy).all() or not np.isfinite(error).all():
        raise ValueError("Training energy or absolute error contains non-finite values.")
    return energy[:, 0], error[:, 0].astype(float) * 1000


def export(output, energy, error, reference_energy=None):
    output.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, len(error) + 1)
    with TemporaryDirectory(prefix=".energy-error-", dir=output) as work:
        tmp = Path(work)
        csv_path = tmp / "absolute_energy_error_online.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["completed_updates", "energy_ewm_Ha", "absolute_energy_error_mHa"])
            writer.writerows(zip(steps, energy, error))
        plot_path = tmp / "absolute_energy_error_online.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        ax.plot(steps, error, label="Online EWM energy error", linewidth=1.4)
        ax.axhline(CHEMICAL_ACCURACY_MHA, color="0.35", linestyle="--", label="1 kcal/mol (1.59 mHa)")
        ax.set(xlabel="Completed training updates", ylabel="Absolute energy error (mHa)")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()
        try:
            fig.savefig(plot_path, dpi=150)
        finally:
            plt.close(fig)

        energy_path = tmp / "energy_online.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        ax.plot(steps, energy, label="Online EWM energy", linewidth=1.4)
        if reference_energy is not None:
            ax.axhline(reference_energy, color="0.35", linestyle="--", label="Reference energy")
        ax.set(xlabel="Completed training updates", ylabel="Energy (Ha)")
        ax.grid(alpha=0.25)
        ax.legend()
        try:
            fig.savefig(energy_path, dpi=150)
        finally:
            plt.close(fig)
        for path in (plot_path, energy_path, csv_path):
            path.replace(output / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training -w directory.")
    parser.add_argument("--watch", action="store_true", help="Refresh periodically until Ctrl-C.")
    parser.add_argument("--interval", type=float, default=60, help="Seconds between reads in watch mode.")
    parser.add_argument("--reference-energy", type=float, help="Reference energy in Ha for the energy plot.")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be a positive finite number.")
    if args.reference_energy is not None and not math.isfinite(args.reference_energy):
        parser.error("--reference-energy must be finite.")
    source = args.run_dir / "training/result.h5"
    output = args.run_dir / "analysis"
    previous_count = None
    previous_error = None
    while True:
        try:
            energy, error = read_metrics(source)
        except (OSError, KeyError) as exc:
            if not args.watch:
                raise
            message = str(exc)
            if message != previous_error:
                print(f"Waiting for metrics in {source}: {message}", flush=True)
                previous_error = message
        else:
            previous_error = None
            if len(error) != previous_count:
                export(output, energy, error, args.reference_energy)
                previous_count = len(error)
                print(f"Step {len(error)}: |E - E_ref| = {error[-1]:.4f} mHa", flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
