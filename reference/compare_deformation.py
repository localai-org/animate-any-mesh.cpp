#!/usr/bin/env python3
"""Compare two 16-frame mesh deformations without relying on a renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from render_motion_points import load_positions


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.reshape(-1)))
    numerator = float(np.linalg.norm((value - reference).reshape(-1)))
    return numerator / max(denominator, np.finfo(np.float64).tiny)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="upstream NPZ or exported GLB")
    parser.add_argument("candidate", type=Path, help="C++ NPZ or exported GLB")
    args = parser.parse_args()

    reference = load_positions(args.reference).astype(np.float64)
    candidate = load_positions(args.candidate).astype(np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: candidate {candidate.shape}, reference {reference.shape}"
        )

    base = reference[0]
    diagonal = float(np.linalg.norm(np.ptp(base, axis=0)))
    frames: list[dict[str, float | int]] = []
    for frame in range(1, reference.shape[0]):
        reference_delta = reference[frame] - reference[0]
        candidate_delta = candidate[frame] - candidate[0]
        reference_norm = float(np.linalg.norm(reference_delta.reshape(-1)))
        candidate_norm = float(np.linalg.norm(candidate_delta.reshape(-1)))
        dot = float(np.vdot(candidate_delta.reshape(-1), reference_delta.reshape(-1)))
        denominator = candidate_norm * reference_norm
        vertex_error = np.linalg.norm(candidate[frame] - reference[frame], axis=1)
        centroid_error = float(
            np.linalg.norm(candidate[frame].mean(axis=0) - reference[frame].mean(axis=0))
        )
        frames.append(
            {
                "frame": frame,
                "position_relative_l2": relative_l2(candidate[frame], reference[frame]),
                "displacement_relative_l2": relative_l2(candidate_delta, reference_delta),
                "displacement_direction_cosine": dot / denominator if denominator else 1.0,
                "displacement_magnitude_ratio": (
                    candidate_norm / reference_norm if reference_norm else 1.0
                ),
                "centroid_error_fraction_of_bbox": centroid_error / max(diagonal, 1.0e-30),
                "maximum_vertex_error_fraction_of_bbox": (
                    float(vertex_error.max()) / max(diagonal, 1.0e-30)
                ),
            }
        )

    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "shape": list(reference.shape),
        "base_position_relative_l2": relative_l2(candidate[0], reference[0]),
        "worst_displacement_relative_l2": max(
            frame["displacement_relative_l2"] for frame in frames
        ),
        "minimum_displacement_direction_cosine": min(
            frame["displacement_direction_cosine"] for frame in frames
        ),
        "frames": frames,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
