"""Summarize a fixed-checkpoint evaluation of one molecular geometry."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from oneqmc.analysis.energy import robust_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("geometry", type=Path)
    args = parser.parse_args()

    training_dir = args.evaluation_dir / "training"
    config = yaml.safe_load((training_dir / "config.yaml").read_text())["args"]
    if not config["test"] or config["mol_batch_size"] != 1 or config["dataset"] != args.geometry.parent.name:
        raise ValueError("Expected fixed-checkpoint evaluation of the supplied single-geometry dataset.")

    with h5py.File(training_dir / "result.h5", "r") as result:
        indices = result["metrics/mol_idx"][:]
        energies = result["metrics/E_loc/mean_elec"][:].astype(np.float64)
    expected = (config["train_steps"], 1)
    if indices.shape != expected or energies.shape != expected:
        raise ValueError(f"Expected {expected} evaluation records, got {indices.shape} and {energies.shape}.")
    if np.any(indices != 0) or not np.isfinite(energies).all():
        raise ValueError("Evaluation contains unexpected geometry IDs or non-finite energies.")

    reference = yaml.safe_load(args.geometry.read_text())["extras"]
    energy = robust_mean(energies[:, 0], delta=1.0).item()
    summary = {
        "dataset": config["dataset"],
        "checkpoint": config["chkpt"],
        "seed": config["seed"],
        "evaluation_steps": len(energies),
        "electron_batch_size": config["electron_batch_size"],
        "energy_Ha": energy,
        "reference_energy_Ha": reference["reference_energy"],
        "absolute_energy_error_mHa": abs(energy - reference["reference_energy"]) * 1000,
    }
    if "reference_energy_std" in reference:
        summary["reference_energy_std_mHa"] = reference["reference_energy_std"] * 1000
    output = args.evaluation_dir / "analysis"
    output.mkdir(exist_ok=True)
    result = json.dumps(summary, indent=2, allow_nan=False)
    (output / "evaluation.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
