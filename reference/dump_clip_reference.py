#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import struct
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


def write_gguf(path: Path, tensors: dict[str, np.ndarray], prompt: str) -> None:
    metadata = text("reference.prompt") + struct.pack("<I", STRING) + text(prompt)
    infos = bytearray()
    offset = 0
    entries = []
    for name, value in tensors.items():
        value = np.ascontiguousarray(value)
        kind = F32 if value.dtype == np.float32 else I32
        dimensions = tuple(reversed(value.shape)) if value.shape else (1,)
        entries.append((name, value, kind, dimensions, offset))
        offset = align(offset + value.nbytes)
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


def safe_state(path: Path):
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return value.get("model_state_dict", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="The object is flying")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    config = CLIPTextConfig(
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
    model = CLIPTextModel(config)
    source = safe_state(args.checkpoint)
    prefix = "clip_text_model."
    clip_state = {name[len(prefix):]: value for name, value in source.items() if name.startswith(prefix)}
    model.load_state_dict(clip_state, strict=True)
    model.eval()

    tokenizer = CLIPTokenizer(
        vocab_file=str(args.tokenizer / "vocab.json"),
        merges_file=str(args.tokenizer / "merges.txt"),
        model_max_length=77,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    token_ids = tokenizer(
        [args.prompt], padding="max_length", truncation=True, max_length=77,
        return_tensors="pt").input_ids
    captured: dict[str, np.ndarray] = {"tokens": token_ids.numpy().astype("<i4")}

    handles = []
    for index, layer in enumerate(model.text_model.encoder.layers):
        def capture(_module, _inputs, output, index=index):
            captured[f"clip.layer.{index}"] = output[0].detach().float().numpy().astype("<f4")
        handles.append(layer.register_forward_hook(capture))
    output = model(token_ids).last_hidden_state
    captured["clip.final"] = output.detach().float().numpy().astype("<f4")
    for handle in handles:
        handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(args.output, captured, args.prompt)
    print(json.dumps({name: list(value.shape) for name, value in captured.items()}, indent=2))


if __name__ == "__main__":
    main()
