#!/usr/bin/env python3
"""Convert the official AnimateAnyMesh checkpoints to one GGUF v3 file.

The loader deliberately uses ``torch.load(weights_only=True)``. PyTorch .pth
files are pickle containers, so disabling that guard would make conversion of
an untrusted checkpoint equivalent to executing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9

UPSTREAM_REVISION = "a0e1d824e59ff4725799cb70511cdd55b30d750b"
MODEL_REVISION = "d3360418409cc51d7bdc0c9633dfb93844113696"
TOKENIZER_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def kv(key: str, kind: int, payload: bytes) -> bytes:
    return gguf_string(key) + struct.pack("<I", kind) + payload


def kv_u32(key: str, value: int) -> bytes:
    return kv(key, GGUF_UINT32, struct.pack("<I", value))


def kv_f32(key: str, value: float) -> bytes:
    return kv(key, GGUF_FLOAT32, struct.pack("<f", value))


def kv_bool(key: str, value: bool) -> bytes:
    return kv(key, GGUF_BOOL, struct.pack("<?", value))


def kv_str(key: str, value: str) -> bytes:
    return kv(key, GGUF_STRING, gguf_string(value))


def kv_strings(key: str, values: list[str]) -> bytes:
    payload = bytearray(struct.pack("<IQ", GGUF_STRING, len(values)))
    for value in values:
        payload += gguf_string(value)
    return kv(key, GGUF_ARRAY, payload)


def align(value: int) -> int:
    return (value + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def state_dict(path: Path):
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise RuntimeError(
            f"safe loading failed for {path}; refusing to retry with weights_only=False"
        ) from exc
    if isinstance(value, dict) and "model_state_dict" in value:
        value = value["model_state_dict"]
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path} does not contain a state dictionary")
    result = {}
    for name, tensor in value.items():
        if not hasattr(tensor, "shape") or not hasattr(tensor, "detach"):
            continue
        clean_name = name.removeprefix("module.")
        result[clean_name] = tensor.detach().cpu()
    if not result:
        raise ValueError(f"{path} contains no tensors")
    return result


def read_tokenizer(directory: Path) -> tuple[list[str], list[str]]:
    with (directory / "vocab.json").open(encoding="utf-8") as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("tokenizer vocab.json must be a non-empty object")
    tokens = [""] * len(mapping)
    for token, index in mapping.items():
        if not isinstance(token, str) or not isinstance(index, int) or not 0 <= index < len(tokens):
            raise ValueError("tokenizer vocabulary IDs must be contiguous integers")
        if tokens[index]:
            raise ValueError("tokenizer vocabulary contains a duplicate ID")
        tokens[index] = token
    if any(token == "" for token in tokens):
        raise ValueError("tokenizer vocabulary IDs are not contiguous")
    with (directory / "merges.txt").open(encoding="utf-8") as stream:
        lines = [line.strip() for line in stream if line.strip()]
    if lines and lines[0].startswith("#version"):
        lines.pop(0)
    return tokens, lines


def choose_type(name: str, shape: tuple[int, ...], ftype: int) -> int:
    if ftype == 0:
        return GGML_TYPE_F32
    fps_parity_weight = name.startswith(("v.pe.", "v.nl."))
    precision_sensitive = fps_parity_weight or any(
        part in name for part in ("norm", "embedding", "basis", "freq_bands")
    )
    return GGML_TYPE_F16 if len(shape) >= 2 and not precision_sensitive else GGML_TYPE_F32


def compact_name(namespace: str, name: str) -> str:
    """Stable <=63-byte GGUF name derived from an upstream state-dict key."""
    if namespace == "rf" and name.startswith("clip_text_model.text_model."):
        result = "c." + name.removeprefix("clip_text_model.text_model.")
    elif namespace == "rf" and name.startswith("backbone.resblocks."):
        result = "r.b." + name.removeprefix("backbone.resblocks.")
    else:
        result = ("v." if namespace == "dvae" else "r.") + name
    replacements = (
        ("encoder.layers.", "l."),
        ("embeddings.", "em."),
        ("token_embedding", "tok"),
        ("position_embedding", "pos"),
        ("final_layer_norm", "fn"),
        ("neighbor_layers", "nl"),
        ("enc_blocks", "eb"),
        ("dec_blocks", "db"),
        ("point_embed", "pe"),
        ("traj_embed", "te"),
        ("decoder_final_ca", "ca"),
        ("mean_fc_x0", "mx0"),
        ("mean_fc_xt", "mxt"),
        ("logvar_fc_xt", "lvxt"),
        ("proj_x0", "px0"),
        ("proj_xt", "pxt"),
        ("fc_query", "fq"),
        ("to_outputs", "out"),
        ("time_embed", "t"),
        ("clip_token_mlp", "ct"),
        ("input_proj", "in"),
        ("output_proj", "out"),
        ("self_attn", "sa"),
        ("layer_norm1", "n1"),
        ("layer_norm2", "n2"),
        ("ln_pre_text", "npt"),
        ("ln_pre", "np"),
        ("ln_post", "no"),
        ("adaln_text", "at"),
        ("adaln_x", "ax"),
        ("norm_q", "nq"),
        ("norm_k", "nk"),
        ("norm_v1", "nv1"),
        ("norm_v2", "nv2"),
        ("norm_v", "nv"),
        ("freq_bands", "freq"),
        (".attention.", ".a."),
        (".attn.", ".a."),
        (".linear.", ".l."),
        (".weight", ".w"),
        (".bias", ".b"),
        (".to_out1", ".o1"),
        (".to_out2", ".o2"),
        (".to_out", ".o"),
        (".to_qkv", ".qkv"),
        (".to_q", ".q"),
        (".to_k", ".k"),
        (".to_v1", ".v1"),
        (".to_v2", ".v2"),
        (".to_v", ".v"),
        (".c_qkv", ".qkv"),
        (".c_proj", ".p"),
        (".c_fc", ".f"),
        (".ffn.", ".f."),
        (".mlp.", ".m."),
    )
    for old, new in replacements:
        result = result.replace(old, new)
    if len(result.encode("utf-8")) >= 64:
        raise ValueError(f"compacted tensor name is still too long: {result}")
    return result


@dataclass(frozen=True)
class TensorInfo:
    name: str
    tensor: object
    ggml_type: int
    dimensions: tuple[int, ...]
    size: int
    offset: int


def tensor_bytes(tensor, ggml_type: int) -> bytes:
    value = tensor.float().contiguous().numpy()
    if ggml_type == GGML_TYPE_F32:
        return value.astype("<f4", copy=False).tobytes(order="C")
    if ggml_type == GGML_TYPE_F16:
        return value.astype("<f2", copy=False).tobytes(order="C")
    raise AssertionError(f"unsupported GGML type {ggml_type}")


def metadata(ftype: int, tokens: list[str], merges: list[str], dvae_hash: str, rf_hash: str) -> list[bytes]:
    prefix = "animate_any_mesh."
    return [
        kv_str("general.architecture", "animate-any-mesh"),
        kv_str("general.name", "AnimateAnyMesh ICCV 2025"),
        kv_u32("general.file_type", ftype),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_str(prefix + "upstream_revision", UPSTREAM_REVISION),
        kv_str(prefix + "model_revision", MODEL_REVISION),
        kv_str(prefix + "tokenizer_revision", TOKENIZER_REVISION),
        kv_str(prefix + "dvae_sha256", dvae_hash),
        kv_str(prefix + "rf_sha256", rf_hash),
        kv_u32(prefix + "frames", 16),
        kv_u32(prefix + "vae.enc_depth", 8),
        kv_u32(prefix + "vae.dec_depth", 8),
        kv_u32(prefix + "vae.width", 256),
        kv_u32(prefix + "vae.latent_width", 32),
        kv_u32(prefix + "vae.output_width", 48),
        kv_u32(prefix + "vae.neighbor_layers", 1),
        kv_u32(prefix + "rf.width", 512),
        kv_u32(prefix + "rf.layers", 12),
        kv_u32(prefix + "rf.heads", 8),
        kv_u32(prefix + "rf.input_channels", 64),
        kv_u32(prefix + "rf.output_channels", 64),
        kv_u32(prefix + "clip.width", 768),
        kv_u32(prefix + "clip.layers", 12),
        kv_u32(prefix + "clip.heads", 12),
        kv_u32(prefix + "clip.context_length", 77),
        kv_f32(prefix + "latent.f0_mean", 0.205227330327034),
        kv_f32(prefix + "latent.f0_std", 9.064417839050293),
        kv_f32(prefix + "latent.ft_mean", 0.95684415102005),
        kv_f32(prefix + "latent.ft_std", 6.397393703460693),
        kv_str("tokenizer.ggml.model", "clip"),
        kv_u32("tokenizer.ggml.bos_token_id", 49406),
        kv_u32("tokenizer.ggml.eos_token_id", 49407),
        kv_bool("tokenizer.ggml.add_bos_token", True),
        kv_bool("tokenizer.ggml.add_eos_token", True),
        kv_strings("tokenizer.ggml.tokens", tokens),
        kv_strings("tokenizer.ggml.merges", merges),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvae", type=Path, required=True)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="directory containing pinned vocab.json and merges.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ftype", type=int, choices=(0, 1), default=1,
                        help="0 = all f32, 1 = f16 matrices with f32 norms/embeddings")
    args = parser.parse_args()

    print("loading checkpoints safely...", flush=True)
    dvae = state_dict(args.dvae)
    rf = state_dict(args.rf)
    required = {
        "dvae.point_embed.mlp.weight": (dvae, "point_embed.mlp.weight"),
        "dvae.to_outputs.weight": (dvae, "to_outputs.weight"),
        "rf.input_proj.weight": (rf, "input_proj.weight"),
        "rf.clip_text_model.text_model.embeddings.token_embedding.weight":
            (rf, "clip_text_model.text_model.embeddings.token_embedding.weight"),
    }
    for display, (source, key) in required.items():
        if key not in source:
            raise KeyError(f"official checkpoint is missing required tensor {display}")

    tokens, merges = read_tokenizer(args.tokenizer)
    all_tensors = [(compact_name("dvae", name), tensor) for name, tensor in dvae.items()]
    all_tensors += [(compact_name("rf", name), tensor) for name, tensor in rf.items()]
    all_tensors.sort(key=lambda item: item[0])
    compact_names = [name for name, _tensor in all_tensors]
    if len(set(compact_names)) != len(compact_names):
        raise ValueError("compact tensor-name mapping contains a collision")

    infos: list[TensorInfo] = []
    offset = 0
    for name, tensor in all_tensors:
        shape = tuple(int(value) for value in tensor.shape)
        ggml_type = choose_type(name, shape, args.ftype)
        item_size = 4 if ggml_type == GGML_TYPE_F32 else 2
        size = int(np.prod(shape, dtype=np.int64)) * item_size
        dimensions = tuple(reversed(shape)) if shape else (1,)
        infos.append(TensorInfo(name, tensor, ggml_type, dimensions, size, offset))
        offset = align(offset + size)

    dvae_hash = sha256(args.dvae)
    rf_hash = sha256(args.rf)
    values = metadata(args.ftype, tokens, merges, dvae_hash, rf_hash)
    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", GGUF_VERSION, len(infos), len(values))
    for value in values:
        header += value
    table = bytearray()
    for info in infos:
        table += gguf_string(info.name)
        table += struct.pack("<I", len(info.dimensions))
        table += struct.pack("<" + "Q" * len(info.dimensions), *info.dimensions)
        table += struct.pack("<IQ", info.ggml_type, info.offset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    print(f"writing {len(infos)} tensors to {args.output}...", flush=True)
    try:
        with temporary.open("wb") as stream:
            stream.write(header)
            stream.write(table)
            stream.write(b"\0" * (align(stream.tell()) - stream.tell()))
            data_start = stream.tell()
            for index, info in enumerate(infos):
                expected = data_start + info.offset
                if stream.tell() != expected:
                    stream.write(b"\0" * (expected - stream.tell()))
                raw = tensor_bytes(info.tensor, info.ggml_type)
                if len(raw) != info.size:
                    raise AssertionError(f"serialized size mismatch for {info.name}")
                stream.write(raw)
                if index % 50 == 0:
                    print(f"  {index:4d}/{len(infos)} {info.name}", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
