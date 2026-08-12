#!/usr/bin/env python3
"""Pack a captured end-to-end NPZ into the small GGUF format used by C++ tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dump_vae_reference import write_gguf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        tensors = {
            "input.vertices": np.asarray(archive["source_vertices"], dtype="<f4"),
            "input.faces": np.asarray(archive["faces"], dtype="<i4"),
            "sample.indices": np.asarray(archive["indices"], dtype="<i4"),
            "sample.epsilon": np.asarray(archive["epsilon"], dtype="<f4"),
            "sample.noise": np.asarray(archive["noise"], dtype="<f4"),
            "expected.encoded": np.asarray(archive["encoded"], dtype="<f4"),
            "expected.rf_normalized": np.asarray(archive["rf_normalized"], dtype="<f4"),
            "expected.latent": np.asarray(archive["latent"], dtype="<f4"),
            "output.positions": np.asarray(archive["positions"], dtype="<f4"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(args.output, tensors)


if __name__ == "__main__":
    main()
