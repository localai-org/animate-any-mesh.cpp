#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch

MAGIC = b"GGUF"
ALIGNMENT = 32
F32 = 0
I32 = 26
STRING = 8


def sample_farthest_points(points, lengths=None, K=50, random_start_point=False):
    """Small, exact squared-distance FPS replacement for the PyTorch3D op."""
    batch, count, _width = points.shape
    if isinstance(K, int):
        counts = [K] * batch
        max_k = K
    else:
        counts = [int(value) for value in K]
        max_k = max(counts)
    indices = torch.full((batch, max_k), -1, dtype=torch.int64, device=points.device)
    selected_points = torch.zeros((batch, max_k, points.shape[-1]), dtype=points.dtype,
                                  device=points.device)
    for b in range(batch):
        valid = int(lengths[b]) if lengths is not None else count
        take = min(counts[b], valid)
        if take == 0:
            continue
        current = int(torch.randint(valid, (1,), device=points.device)) if random_start_point else 0
        closest = torch.full((valid,), torch.inf, dtype=points.dtype, device=points.device)
        for i in range(take):
            indices[b, i] = current
            selected_points[b, i] = points[b, current]
            distance = ((points[b, :valid] - points[b, current]) ** 2).sum(-1)
            closest = torch.minimum(closest, distance)
            current = int(torch.argmax(closest))
    return selected_points, indices


def install_pytorch3d_stub() -> None:
    package = types.ModuleType("pytorch3d")
    ops = types.ModuleType("pytorch3d.ops")
    ops.sample_farthest_points = sample_farthest_points
    package.ops = ops
    sys.modules["pytorch3d"] = package
    sys.modules["pytorch3d.ops"] = ops


def text(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def write_gguf(path: Path, tensors: dict[str, np.ndarray]) -> None:
    metadata = text("reference.kind") + struct.pack("<I", STRING) + text("dymeshvae")
    entries = []
    offset = 0
    for name, value in tensors.items():
        value = np.ascontiguousarray(value)
        kind = I32 if value.dtype == np.int32 else F32
        dimensions = tuple(reversed(value.shape)) if value.shape else (1,)
        entries.append((name, value, kind, dimensions, offset))
        offset = align(offset + value.nbytes)
    infos = bytearray()
    for name, value, kind, dimensions, offset in entries:
        infos += text(name)
        infos += struct.pack("<I", len(dimensions))
        infos += struct.pack("<" + "Q" * len(dimensions), *dimensions)
        infos += struct.pack("<IQ", kind, offset)
    header = MAGIC + struct.pack("<IQQ", 3, len(entries), 1) + metadata
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(infos)
        stream.write(b"\0" * (align(stream.tell()) - stream.tell()))
        data_start = stream.tell()
        for _name, value, _kind, _dimensions, offset in entries:
            stream.write(b"\0" * (data_start + offset - stream.tell()))
            stream.write(value.tobytes())


def record(output: dict[str, np.ndarray], name: str, value: torch.Tensor) -> None:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    output[name] = array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--trajectories", type=int, default=8)
    args = parser.parse_args()

    install_pytorch3d_stub()
    from networks.dymeshvae import DyMeshVAE, replace_negative_indices

    # Upstream explicitly disables the math SDPA kernel to force CUDA flash.
    # The synchronized attention concatenates two value streams (Ev=2*Eq),
    # which PyTorch's flash kernel cannot handle. For the portable reference,
    # retain the same SDPA equation while allowing its exact math backend.
    @contextlib.contextmanager
    def math_sdpa(**_kwargs):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    torch.backends.cuda.sdp_kernel = math_sdpa

    torch.set_grad_enabled(False)
    torch.manual_seed(1234)
    config = dict(enc_depth=8, dec_depth=8, dim=256, output_dim=48,
                  latent_dim=32, T=16, num_traj=args.trajectories, n_layers=1)
    model = DyMeshVAE(**config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=True)
    model.eval()

    angle = torch.linspace(0.0, 2.0 * torch.pi, args.points + 1)[:-1]
    pc0 = torch.stack((0.7 * torch.cos(angle), 0.4 * torch.sin(angle),
                       torch.linspace(-0.8, 0.8, args.points)), dim=-1)[None]
    pc = pc0[:, None].repeat(1, 16, 1, 1)
    valid_mask = torch.ones((1, args.points), dtype=torch.bool)
    adjacency = torch.zeros((1, args.points, args.points), dtype=torch.bool)
    for index in range(args.points):
        adjacency[0, index, (index - 1) % args.points] = True
        adjacency[0, index, (index + 1) % args.points] = True

    tensors: dict[str, np.ndarray] = {}
    record(tensors, "input.pc0", pc0)
    record(tensors, "input.adjacency", adjacency.float())

    relative = (pc - pc[:, :1]).permute(0, 2, 1, 3).flatten(2, 3)
    point = model.point_embed(pc0)
    trajectory = model.traj_embed(relative)
    record(tensors, "enc.point", point)
    record(tensors, "enc.trajectory", trajectory)
    for layer in model.neighbor_layers:
        neighbor_residual = layer(point, key=point, value=point, mask=adjacency)
        record(tensors, "enc.neighbor.residual", neighbor_residual)
        point = point + neighbor_residual
    point_original = point
    trajectory_original = trajectory
    record(tensors, "enc.neighbor.output", point_original)

    valid_length = valid_mask.sum(-1)
    _, indices = sample_farthest_points(point, lengths=valid_length, K=args.trajectories)
    indices = replace_negative_indices(indices, valid_length)
    tensors["enc.indices"] = indices.cpu().numpy().astype("<i4")
    gather = indices.unsqueeze(-1).expand(-1, -1, point.shape[-1])
    point = torch.gather(point, 1, gather)
    trajectory = torch.gather(trajectory, 1, gather)
    record(tensors, "enc.sampled.point", point)
    record(tensors, "enc.sampled.trajectory", trajectory)

    for index, (attention, feed_forward) in enumerate(model.enc_blocks):
        point_residual, trajectory_residual = attention(
            q_stream=point, k_stream=point_original,
            v1_stream=point_original, v2_stream=trajectory_original)
        point = point + point_residual
        trajectory = trajectory + trajectory_residual
        record(tensors, f"enc.block.{index}.attention.point", point)
        record(tensors, f"enc.block.{index}.attention.trajectory", trajectory)
        point_residual, trajectory_residual = feed_forward(point, trajectory)
        point = point + point_residual
        trajectory = trajectory + trajectory_residual
        record(tensors, f"enc.block.{index}.point", point)
        record(tensors, f"enc.block.{index}.trajectory", trajectory)

    latent_x0 = model.mean_fc_x0(point)
    mean = model.mean_fc_xt(trajectory)
    log_variance = torch.clamp(model.logvar_fc_xt(trajectory), -30.0, 20.0)
    standard_deviation = torch.exp(0.5 * log_variance)
    epsilon = torch.randn(mean.shape)
    latent_xt = mean + standard_deviation * epsilon
    latent = torch.cat((latent_x0, latent_xt), dim=-1)
    record(tensors, "enc.latent.x0", latent_x0)
    record(tensors, "enc.latent.mean", mean)
    record(tensors, "enc.latent.logvar", log_variance)
    record(tensors, "enc.latent.epsilon", epsilon)
    record(tensors, "enc.latent.xt", latent_xt)
    record(tensors, "enc.latent", latent)

    decoded_x0 = model.proj_x0(latent_x0)
    decoded_xt = model.proj_xt(latent_xt)
    record(tensors, "dec.projected.x0", decoded_x0)
    record(tensors, "dec.projected.xt", decoded_xt)
    for index, (attention, feed_forward) in enumerate(model.dec_blocks):
        point_residual, trajectory_residual = attention(
            q_stream=decoded_x0, k_stream=decoded_x0,
            v1_stream=decoded_x0, v2_stream=decoded_xt)
        decoded_x0 = decoded_x0 + point_residual
        decoded_xt = decoded_xt + trajectory_residual
        record(tensors, f"dec.block.{index}.attention.x0", decoded_x0)
        record(tensors, f"dec.block.{index}.attention.xt", decoded_xt)
        point_residual, trajectory_residual = feed_forward(decoded_x0, decoded_xt)
        decoded_x0 = decoded_x0 + point_residual
        decoded_xt = decoded_xt + trajectory_residual
        record(tensors, f"dec.block.{index}.x0", decoded_x0)
        record(tensors, f"dec.block.{index}.xt", decoded_xt)

    query = model.fc_query(point_original)
    cross_attention = model.decoder_final_ca(query, key=decoded_x0, value=decoded_xt)
    offsets = model.to_outputs(cross_attention)
    output = offsets.view(1, args.points, -1, 3).permute(0, 2, 1, 3) + pc0[:, None]
    record(tensors, "dec.query", query)
    record(tensors, "dec.cross_attention", cross_attention)
    record(tensors, "dec.offsets", offsets)
    record(tensors, "output", output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(args.output, tensors)
    print(json.dumps({name: list(value.shape) for name, value in tensors.items()}, indent=2))


if __name__ == "__main__":
    main()
