#!/usr/bin/env python3
"""Emit deterministic, layer-addressable MMDiT reference tensors."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

MAGIC = b"GGUF"
ALIGNMENT = 32
F32 = 0
I32 = 26
STRING = 8


def text(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def write_gguf(path: Path, tensors: dict[str, np.ndarray]) -> None:
    metadata = text("reference.kind") + struct.pack("<I", STRING) + text("dymesh-mmdit")
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
    output[name] = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream))
    import networks.transformer as upstream
    from networks.util import timestep_embedding

    torch.set_grad_enabled(False)
    torch.manual_seed(2025)
    clip_config = CLIPTextConfig(
        vocab_size=49408, hidden_size=768, intermediate_size=3072,
        projection_dim=768, num_hidden_layers=12, num_attention_heads=12,
        max_position_embeddings=77, hidden_act="quick_gelu", layer_norm_eps=1.0e-5,
        attention_dropout=0.0, bos_token_id=49406, eos_token_id=49407, pad_token_id=1,
    )
    tokenizer = CLIPTokenizer(
        vocab_file=str(args.tokenizer / "vocab.json"),
        merges_file=str(args.tokenizer / "merges.txt"), model_max_length=77,
        bos_token="<|startoftext|>", eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    original_model_loader = upstream.CLIPTextModel.from_pretrained
    original_tokenizer_loader = upstream.CLIPTokenizer.from_pretrained
    upstream.CLIPTextModel.from_pretrained = staticmethod(lambda *_a, **_k: CLIPTextModel(clip_config))
    upstream.CLIPTokenizer.from_pretrained = staticmethod(lambda *_a, **_k: tokenizer)
    try:
        model = upstream.DyMeshMMDiT(
            device="cpu", dtype=torch.float32, input_channels=64, output_channels=64,
            width=512, layers=12, heads=8, use_flash2=False)
    finally:
        upstream.CLIPTextModel.from_pretrained = original_model_loader
        upstream.CLIPTokenizer.from_pretrained = original_tokenizer_loader

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model_state_dict", state)
    state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    prompts = ["The object is flying", ""]
    ids = tokenizer(prompts, padding="max_length", truncation=True, max_length=77,
                    return_tensors="pt").input_ids
    batch = len(prompts)
    positions = torch.arange(args.tokens * 64, dtype=torch.float32).reshape(1, args.tokens, 64)
    x = torch.sin(positions * 0.03125).repeat(batch, 1, 1)
    x[1] = torch.cos(positions[0] * 0.021)
    times = torch.tensor([0.375, 0.375], dtype=torch.float32)

    tensors: dict[str, np.ndarray] = {
        "tokens": ids.numpy().astype("<i4"),
    }
    record(tensors, "input.x", x)
    record(tensors, "input.t", times)
    clip = model.clip_text_model(ids).last_hidden_state
    record(tensors, "clip.final", clip)
    time_raw = timestep_embedding(times, model.backbone.width)
    time = model.time_embed(time_raw)
    record(tensors, "time.raw", time_raw)
    record(tensors, "time.projected", time)
    text_features = model.clip_token_mlp(clip)
    text_features[1].zero_()
    record(tensors, "text.projected", text_features)
    x_features = model.ln_pre(model.input_proj(x))
    text_features = model.ln_pre_text(text_features)
    record(tensors, "x.initial", x_features)
    record(tensors, "text.initial", text_features)

    for index, block in enumerate(model.backbone.resblocks):
        count = x_features.shape[1]
        norm_x, gate_ax, shift_fx, scale_fx, gate_fx = block.adaln_x(x_features, emb=time)
        norm_text, gate_at, shift_ft, scale_ft, gate_ft = block.adaln_text(text_features, emb=time)
        attention = block.attn(torch.cat((norm_x, norm_text), dim=1))
        x_features = x_features + gate_ax[:, None] * attention[:, :count]
        text_features = text_features + gate_at[:, None] * attention[:, count:]
        record(tensors, f"block.{index}.attention.x", x_features)
        record(tensors, f"block.{index}.attention.text", text_features)
        norm_x = block.ln_x(x_features) * (1 + scale_fx[:, None]) + shift_fx[:, None]
        norm_text = block.ln_text(text_features) * (1 + scale_ft[:, None]) + shift_ft[:, None]
        feed_forward = block.mlp(torch.cat((norm_x, norm_text), dim=1))
        x_features = x_features + gate_fx[:, None] * feed_forward[:, :count]
        text_features = text_features + gate_ft[:, None] * feed_forward[:, count:]
        record(tensors, f"block.{index}.x", x_features)
        record(tensors, f"block.{index}.text", text_features)

    final_norm = model.ln_post(x_features)
    output = model.output_proj(final_norm)
    record(tensors, "final.norm", final_norm)
    record(tensors, "output", output)

    # A compact fixed-grid midpoint fixture validates the ODE wrapper, CFG,
    # forced f0 channels, and final channel splice independently of any RNG.
    sample_tokens = min(args.tokens, 4)
    f0 = torch.sin(torch.arange(sample_tokens * 32, dtype=torch.float32).reshape(
        1, sample_tokens, 32) * 0.07)
    noise = torch.cos(torch.arange(sample_tokens * 64, dtype=torch.float32).reshape(
        1, sample_tokens, 64) * 0.043)
    state = noise.clone()
    sample_steps = 4
    delta = 1.0 / (sample_steps - 1)

    def velocity(value: torch.Tensor, time: float) -> torch.Tensor:
        doubled = value.repeat(2, 1, 1)
        doubled[:, :, :32] = f0.repeat(2, 1, 1)
        timestep = torch.full((2,), time, dtype=torch.float32)
        time_features = model.time_embed(timestep_embedding(timestep, model.backbone.width))
        conditioned_text = model.clip_token_mlp(clip).clone()
        conditioned_text[1].zero_()
        prediction = model._forward_cogx(doubled, conditioned_text, time_features)
        return prediction[1:2] + 3.0 * (prediction[:1] - prediction[1:2])

    for step in range(sample_steps - 1):
        time = step * delta
        start_velocity = velocity(state, time)
        midpoint = state + 0.5 * delta * start_velocity
        state = state + delta * velocity(midpoint, time + 0.5 * delta)
    state[:, :, :32] = f0
    record(tensors, "sample.f0", f0)
    record(tensors, "sample.noise", noise)
    record(tensors, "sample.output", state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(args.output, tensors)
    print(json.dumps({name: list(value.shape) for name, value in tensors.items()}, indent=2))


if __name__ == "__main__":
    main()
