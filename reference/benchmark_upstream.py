#!/usr/bin/env python3
"""Headless benchmark of the published AnimateAnyMesh inference path.

Rendering is deliberately omitted. The model classes, checkpoints, two VAE
encoder calls made by ``just_encode``/``just_decode``, CLIP calls in every RF
evaluation, midpoint ODE solver, and default F32 precision are upstream code.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import trimesh
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


def install_pytorch3d_stub(exact_fps: bool = False) -> None:
    """Install a conservative FPS substitute for unsupported Blackwell wheels.

    FPS only chooses token indices; returning the first K indices preserves all
    neural tensor shapes and makes the reference slightly faster than its
    compiled PyTorch3D path.
    """

    def sample_farthest_points(points, lengths=None, K=50, random_start_point=False):
        batch, count, width = points.shape
        counts = [K] * batch if isinstance(K, int) else [int(value) for value in K]
        maximum = max(counts)
        indices = torch.full((batch, maximum), -1, dtype=torch.int64, device=points.device)
        selected = torch.zeros((batch, maximum, width), dtype=points.dtype, device=points.device)
        for batch_index, requested in enumerate(counts):
            valid = int(lengths[batch_index]) if lengths is not None else count
            take = min(requested, valid)
            if take == 0:
                continue
            if not exact_fps:
                current = torch.arange(take, dtype=torch.int64, device=points.device)
                indices[batch_index, :take] = current
                selected[batch_index, :take] = points[batch_index, :take]
                continue
            current = (
                int(torch.randint(valid, (1,), device=points.device))
                if random_start_point else 0
            )
            closest = torch.full(
                (valid,), torch.inf, dtype=points.dtype, device=points.device
            )
            for sample_index in range(take):
                indices[batch_index, sample_index] = current
                selected[batch_index, sample_index] = points[batch_index, current]
                distance = (
                    (points[batch_index, :valid] - points[batch_index, current]) ** 2
                ).sum(-1)
                closest = torch.minimum(closest, distance)
                current = int(torch.argmax(closest))
        return selected, indices

    package = types.ModuleType("pytorch3d")
    operations = types.ModuleType("pytorch3d.ops")
    operations.sample_farthest_points = sample_farthest_points
    package.ops = operations
    sys.modules["pytorch3d"] = package
    sys.modules["pytorch3d.ops"] = operations


def load_scene(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    scene = trimesh.load(path, force="scene", process=False)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        current_vertices = trimesh.transform_points(
            np.asarray(geometry.vertices, dtype=np.float64), transform
        ).astype(np.float32)
        current_faces = np.asarray(geometry.faces, dtype=np.int64)
        vertices.append(current_vertices)
        faces.append(current_faces + offset)
        offset += len(current_vertices)
    if not vertices:
        raise ValueError(f"no triangle mesh in {path}")
    return torch.from_numpy(np.concatenate(vertices)), torch.from_numpy(np.concatenate(faces))


def safe_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model_state_dict", state)
    return {name.removeprefix("module."): value for name, value in state.items()}


def make_rf_model(transformer_module, checkpoint: Path, tokenizer_dir: Path, device):
    clip_config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1.0e-5,
        attention_dropout=0.0,
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=1,
    )
    tokenizer = CLIPTokenizer(
        vocab_file=str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt"),
        model_max_length=77,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    original_model_loader = transformer_module.CLIPTextModel.from_pretrained
    original_tokenizer_loader = transformer_module.CLIPTokenizer.from_pretrained
    transformer_module.CLIPTextModel.from_pretrained = staticmethod(
        lambda *_args, **_kwargs: CLIPTextModel(clip_config)
    )
    transformer_module.CLIPTokenizer.from_pretrained = staticmethod(
        lambda *_args, **_kwargs: tokenizer
    )
    try:
        model = transformer_module.DyMeshMMDiT(
            device=device,
            dtype=torch.float32,
            input_channels=64,
            output_channels=64,
            width=512,
            layers=12,
            heads=8,
            use_flash2=True,
        )
    finally:
        transformer_module.CLIPTextModel.from_pretrained = original_model_loader
        transformer_module.CLIPTokenizer.from_pretrained = original_tokenizer_loader
    model.load_state_dict(safe_state(checkpoint), strict=True)
    return model.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dvae", type=Path, required=True)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--prompt", default="The object is flying")
    parser.add_argument(
        "--trajectories", type=int, default=0,
        help="trajectory count; 0 uses upstream max(512, merged vertices / 8)",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="wrap the unchanged upstream VAE and RF modules with torch.compile",
    )
    parser.add_argument(
        "--exact-fps",
        action="store_true",
        help="use exact pure-PyTorch farthest-point sampling instead of the fast index stub",
    )
    parser.add_argument(
        "--faithful-seed-order",
        action="store_true",
        help=(
            "seed once before constructing the models, matching upstream test_drive.py; "
            "requires --warmups 0 --runs 1"
        ),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        help="optionally save normalized output positions and topology for visual comparison",
    )
    args = parser.parse_args()
    if args.trajectories < 0 or args.steps < 2 or args.warmups < 0 or args.runs <= 0:
        parser.error("invalid trajectories, steps, warmups, or runs")
    if args.faithful_seed_order and (args.warmups != 0 or args.runs != 1):
        parser.error("--faithful-seed-order requires --warmups 0 --runs 1")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the upstream GPU benchmark")
    install_pytorch3d_stub(args.exact_fps)
    sys.path.insert(0, str(args.upstream))
    import networks.dymeshvae as dymeshvae_module
    import networks.transformer as transformer_module
    from diffusion.rf_diffusion import rf_sample
    from utils.mesh_utils import get_adjacency_matrix, merge_identical_vertices_with_indices

    captured: dict[str, torch.Tensor] = {}
    original_replace_indices = dymeshvae_module.replace_negative_indices

    def capture_indices(indices, valid_length):
        result = original_replace_indices(indices, valid_length)
        captured.setdefault("indices", result.detach().cpu())
        captured.setdefault("cuda_state_after_indices", torch.cuda.get_rng_state().cpu())
        return result

    def capture_epsilon(distribution):
        epsilon = torch.randn(distribution.mean.shape)
        captured.setdefault("epsilon", epsilon.detach().clone())
        return distribution.mean + distribution.std * epsilon.to(device=distribution.mean.device)

    dymeshvae_module.replace_negative_indices = capture_indices
    dymeshvae_module.DiagonalGaussianDistribution.sample = capture_epsilon
    DyMeshVAE = dymeshvae_module.DyMeshVAE

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    if args.faithful_seed_order:
        # test_drive.py seeds before model construction. Although checkpoint
        # loading replaces the initialized weights, those initializers advance
        # the CPU generator later used for the VAE posterior epsilon.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    load_started = time.perf_counter()
    vae = DyMeshVAE(
        enc_depth=8,
        dec_depth=8,
        dim=256,
        output_dim=48,
        latent_dim=32,
        T=16,
        num_traj=args.trajectories,
        n_layers=1,
    ).to(device)
    vae.load_state_dict(safe_state(args.dvae), strict=True)
    vae.eval()
    rf = make_rf_model(transformer_module, args.rf, args.tokenizer, device)
    cuda_state_after_model = torch.cuda.get_rng_state().cpu()
    if args.compile:
        vae = torch.compile(vae)
        rf = torch.compile(rf)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    mesh_started = time.perf_counter()
    raw_vertices, raw_faces = load_scene(args.mesh)
    merged_vertices, merged_faces, _ = merge_identical_vertices_with_indices(
        [raw_vertices], [raw_faces]
    )
    source_vertices = merged_vertices.clone()
    center = (merged_vertices.max(dim=0).values + merged_vertices.min(dim=0).values) / 2
    vertices = merged_vertices - center
    vertices = vertices / (vertices.abs().max() + 1.0e-8)
    vertex_count = int(vertices.shape[0])
    if args.trajectories == 0:
        args.trajectories = max(512, vertex_count // 8)
    faces_max_length = min(int(vertex_count * 2.5), int(merged_faces.shape[0]))
    faces = merged_faces[:faces_max_length]
    vertices = vertices[None, None].to(device).repeat(1, 16, 1, 1)
    faces = faces[None].to(device)
    valid_mask = ~(vertices.permute(0, 2, 1, 3).flatten(2, 3) == 0.0).all(dim=-1)
    valid_length = valid_mask.sum(dim=-1)
    adjacency = get_adjacency_matrix(vertices[:, 0], faces, valid_length)
    torch.cuda.synchronize()
    mesh_preprocess_seconds = time.perf_counter() - mesh_started

    def inference() -> torch.Tensor:
        captured.clear()
        if not args.faithful_seed_order:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        encoded = vae(
            vertices,
            vertices[:, 0],
            faces=faces,
            valid_mask=valid_mask,
            adj_matrix=adjacency,
            num_traj=args.trajectories,
            just_encode=True,
        )
        captured["encoded"] = encoded.detach().cpu()
        f0_mean, f0_std = 0.205227330327034, 9.064417839050293
        ft_mean, ft_std = 0.95684415102005, 6.397393703460693
        normalized = torch.cat(
            [
                (encoded[:, :, :32] - f0_mean) / f0_std,
                (encoded[:, :, 32:] - ft_mean) / ft_std,
            ],
            dim=-1,
        )
        f0 = normalized[:, :, :32]
        original_randn = torch.randn

        def capture_noise(*shape, **kwargs):
            captured["cuda_state_before_noise"] = torch.cuda.get_rng_state().cpu()
            noise = original_randn(*shape, **kwargs)
            captured["noise"] = noise.detach().cpu()
            return noise

        torch.randn = capture_noise
        try:
            samples = rf_sample(
                model=rf,
                shape=normalized.shape,
                steps=args.steps,
                model_kwargs={"texts": [args.prompt]},
                guidance_scale=args.guidance,
                device=device,
                f0=f0,
            )
        finally:
            torch.randn = original_randn
        captured["rf_normalized"] = samples.detach().cpu()
        samples = torch.cat(
            [samples[:, :, :32] * f0_std + f0_mean, samples[:, :, 32:] * ft_std + ft_mean],
            dim=-1,
        )
        captured["latent"] = samples.detach().cpu()
        return vae(
            vertices,
            vertices[:, 0],
            samples=samples,
            faces=faces,
            valid_mask=valid_mask,
            adj_matrix=adjacency,
            num_traj=args.trajectories,
            just_decode=True,
        )

    warmup_elapsed: list[float] = []
    for warmup in range(args.warmups):
        torch.cuda.synchronize()
        warmup_started = time.perf_counter()
        inference()
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started
        warmup_elapsed.append(warmup_seconds)
        print(
            f"warmup {warmup + 1} seconds {warmup_seconds:.6f}",
            file=sys.stderr,
            flush=True,
        )

    elapsed: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    result = None
    for run in range(args.runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = inference()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        elapsed.append(seconds)
        print(f"run {run + 1} seconds {seconds:.6f}", file=sys.stderr, flush=True)

    assert result is not None
    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_npz,
            positions=result[0].float().cpu().numpy(),
            faces=faces[0].cpu().numpy(),
            source_vertices=source_vertices.cpu().numpy(),
            indices=captured["indices"][0].numpy(),
            epsilon=captured["epsilon"][0].numpy(),
            noise=captured["noise"][0].numpy(),
            encoded=captured["encoded"][0].numpy(),
            rf_normalized=captured["rf_normalized"][0].numpy(),
            latent=captured["latent"][0].numpy(),
            cuda_state_after_model=cuda_state_after_model.numpy(),
            cuda_state_after_indices=captured["cuda_state_after_indices"].numpy(),
            cuda_state_before_noise=captured["cuda_state_before_noise"].numpy(),
        )
    report = {
        "implementation": "upstream-pytorch-headless",
        "execution": "compile" if args.compile else "eager",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "precision": "f32",
        "vertices": vertex_count,
        "faces": int(faces.shape[1]),
        "trajectories": args.trajectories,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "seed_order": (
            "upstream test_drive.py (before model construction)"
            if args.faithful_seed_order
            else "reset immediately before inference"
        ),
        "runs_seconds": elapsed,
        "median_seconds": statistics.median(elapsed),
        "warmup_seconds": warmup_elapsed,
        "model_load_seconds": model_load_seconds,
        "mesh_preprocess_seconds": mesh_preprocess_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
        "output_checksum": float(result.float().sum().item()),
        "cuda_rng_state_bytes": int(cuda_state_after_model.numel()),
        "fps": (
            "exact pure-PyTorch"
            if args.exact_fps
            else "deterministic index stub (conservatively favors upstream)"
        ),
        "rendering": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
