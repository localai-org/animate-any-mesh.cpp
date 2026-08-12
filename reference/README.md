# Python reference environment

The production build is entirely Nix/C++, but the original implementation has
CUDA-, Blender-, and PyTorch3D-specific dependencies. Its validation environment
therefore lives in a separate pinned container.

Build it:

```sh
docker build -t aam-reference:2.6 reference
```

The default PyTorch 2.6/CUDA 12.4 image exactly matches upstream but cannot run
CUDA kernels on Blackwell (`sm_120`). For those GPUs, build the same environment
on a CUDA 12.8 PyTorch base (2.7.1 or newer):

```sh
docker build -t aam-reference:cu128 reference \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
```

The upstream source is pinned to commit
`a0e1d824e59ff4725799cb70511cdd55b30d750b`; mount that checkout read-only at
`/opt/AnimateAnyMesh`. Mount checkpoints read-only and dumps writable:

```sh
docker run --rm --gpus all \
  -v "$PWD/../AnimateAnyMesh:/opt/AnimateAnyMesh:ro" \
  -v "$PWD/checkpoints:/checkpoints:ro" \
  -v "$PWD:/work" aam-reference:2.6 \
  reference/dump_clip_reference.py \
    --checkpoint /checkpoints/rf_epoch_f.pth \
    --tokenizer /checkpoints/tokenizer --output /work/dumps/clip.gguf
```

The other fixtures are `reference/dump_vae_reference.py` and
`reference/dump_rf_reference.py`; run each with `--help` for its explicit
arguments. The VAE dumper replaces only PyTorch3D's farthest-point kernel with
an equivalent pure-PyTorch implementation. This makes small layer-parity
fixtures work on CPU too; the neural layers remain the unmodified upstream
classes. All dumpers use `torch.load(..., weights_only=True)`.

For a rendering-free performance comparison, run
`reference/benchmark_upstream.py`. It preserves the upstream F32 models,
duplicate VAE encode during `just_decode`, repeated CLIP encoding during RF,
and midpoint solver. It replaces only unavailable PyTorch3D FPS with a cheaper
deterministic selector, which conservatively favors upstream:

```sh
docker run --rm --device nvidia.com/gpu=all \
  -v "$PWD/../AnimateAnyMesh:/opt/AnimateAnyMesh:ro" \
  -v "$PWD/checkpoints:/checkpoints:ro" \
  -v "$PWD:/work:ro" aam-reference:cu128 \
  /work/reference/benchmark_upstream.py \
    --upstream /opt/AnimateAnyMesh \
    --dvae /checkpoints/dvae_f.pth --rf /checkpoints/rf_epoch_f.pth \
    --tokenizer /checkpoints/tokenizer --mesh /opt/AnimateAnyMesh/examples/dragon.glb
```

Add `--compile` to wrap the unchanged VAE and RF modules with
`torch.compile`. On the pinned upstream code this succeeds, but Dynamo reports
graph breaks in tokenizer Unicode handling and in the deterministic FPS stub,
so it is a partial rather than full-graph compilation. The JSON report labels
the execution mode and records the lazy-compilation warm-up separately from
the measured runs.

Use `--faithful-seed-order --warmups 0 --runs 1` for an output fixture that
matches `test_drive.py` rather than a repeatable benchmark run. Upstream seeds
before constructing its CUDA RF model, and those discarded initial weights
advance the Philox stream used for inference.

For an end-to-end deformation comparison, capture upstream tensors with
`--output-npz`, pack them with `pack_full_reference.py`, and run
`aam-full-parity` with `AAM_FULL_REFERENCE` and `AAM_TEST_GGUF`. The test can
export matched 10 FPS GLBs through `AAM_FULL_OUTPUT_PREFIX`; it reports the
relative error, direction cosine, and magnitude ratio of every frame's vertex
displacements. `compare_deformation.py` provides the same geometric report for
any two exported GLBs:

```sh
python reference/compare_deformation.py upstream.glb cpp.glb
```

Set `AAM_TEST_PRODUCTION_INDICES=1` to additionally compare the trajectory
indices that production farthest-point sampling chooses. This is important:
the RF is permutation-equivariant, but its seeded noise is paired with tokens
by FPS order, so a numerically reordered trajectory set selects a different
animation even when almost all selected vertices are the same.
