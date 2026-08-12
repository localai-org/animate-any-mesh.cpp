#!/usr/bin/env python3
"""Render a dependency-light point-cloud contact sheet from NPZ or AAM GLB motion."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_glb(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF" or struct.unpack_from("<I", data, 4)[0] != 2:
        raise ValueError("input is not a binary glTF 2 file")
    declared = struct.unpack_from("<I", data, 8)[0]
    if declared != len(data):
        raise ValueError("GLB length does not match its header")
    offset = 12
    document = None
    binary = None
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + length
        if end > len(data):
            raise ValueError("GLB chunk exceeds the file")
        if kind == 0x4E4F534A:
            document = json.loads(data[offset:end].decode("utf-8"))
        elif kind == 0x004E4942:
            binary = memoryview(data)[offset:end]
        offset = end
    if document is None or binary is None:
        raise ValueError("GLB lacks JSON or binary data")

    def vec3(accessor_index: int) -> np.ndarray:
        accessor = document["accessors"][accessor_index]
        if accessor["componentType"] != 5126 or accessor["type"] != "VEC3":
            raise ValueError("motion accessor is not an F32 VEC3")
        view = document["bufferViews"][accessor["bufferView"]]
        start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        count = int(accessor["count"])
        stride = int(view.get("byteStride", 12))
        if count < 1 or start < 0 or stride < 12 or start + (count - 1) * stride + 12 > len(binary):
            raise ValueError("motion accessor exceeds the binary chunk")
        return np.ndarray(
            (count, 3), dtype="<f4", buffer=binary, offset=start, strides=(stride, 4)
        ).copy()

    primitive = document["meshes"][0]["primitives"][0]
    base = vec3(primitive["attributes"]["POSITION"])
    frames = [base]
    for target in primitive.get("targets", []):
        delta = vec3(target["POSITION"])
        if delta.shape != base.shape:
            raise ValueError("morph target does not match the base positions")
        frames.append(base + delta)
    return np.stack(frames)


def load_positions(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".glb":
        positions = load_glb(path)
    else:
        with np.load(path, allow_pickle=False) as archive:
            positions = np.asarray(archive["positions"], dtype=np.float32)
    if positions.ndim != 3 or positions.shape[0] < 2 or positions.shape[2] != 3:
        raise ValueError("positions must have shape [frames, vertices, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("positions contain NaN or infinity")
    return positions


def render(positions: np.ndarray, output: Path, view: str, tile: int) -> None:
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[view]
    columns = 4
    rows = (positions.shape[0] + columns - 1) // columns
    sheet = np.full((rows * tile, columns * tile, 3), 22, dtype=np.uint8)
    base = positions[0]
    center = (base.min(0) + base.max(0)) * 0.5
    span = max(float(np.ptp(base[:, axes[0]])), float(np.ptp(base[:, axes[1]])), 1.0e-6)
    scale = (tile - 16) / (span * 1.12)
    depth = base[:, axes[2]]
    depth_span = max(float(np.ptp(depth)), 1.0e-6)
    shade = np.clip(145 + 100 * (depth - depth.min()) / depth_span, 0, 255).astype(np.uint8)
    for frame_index, frame in enumerate(positions):
        x = np.rint((frame[:, axes[0]] - center[axes[0]]) * scale + tile / 2).astype(int)
        y = np.rint(-(frame[:, axes[1]] - center[axes[1]]) * scale + tile / 2).astype(int)
        valid = (x >= 1) & (x < tile - 1) & (y >= 1) & (y < tile - 1)
        x, y, colors = x[valid], y[valid], shade[valid]
        row, column = divmod(frame_index, columns)
        x += column * tile
        y += row * tile
        for dx, dy in ((0, 0), (1, 0), (0, 1)):
            sheet[y + dy, x + dx, 0] = colors
            sheet[y + dy, x + dx, 1] = np.minimum(
                colors.astype(np.uint16) + 18, 255
            ).astype(np.uint8)
            sheet[y + dy, x + dx, 2] = 255
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode("ascii"))
        stream.write(sheet.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--view", choices=("xy", "xz", "yz"), default="xz")
    parser.add_argument("--tile", type=int, default=320)
    args = parser.parse_args()
    if args.tile < 64 or args.tile > 2048:
        parser.error("tile must be in [64, 2048]")
    render(load_positions(args.input), args.output, args.view, args.tile)


if __name__ == "__main__":
    main()
