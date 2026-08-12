# animate-any-mesh.cpp

A memory-safe C++23 inference implementation of
[AnimateAnyMesh](https://github.com/JarrentWu1031/AnimateAnyMesh), running on
GGML's dynamically selected CPU kernels or Vulkan backend. It implements the
complete published inference path: GLB preprocessing, CLIP tokenization/text
encoding, DyMeshVAE encoding, text-conditioned rectified-flow sampling,
DyMeshVAE decoding, and animated morph-target GLB export. Python is used only
to convert the original trusted checkpoints and to generate validation data.

The implementation is pinned to:

- AnimateAnyMesh commit `a0e1d824e59ff4725799cb70511cdd55b30d750b`
- official model revision `d3360418409cc51d7bdc0c9633dfb93844113696`
- Stable Diffusion CLIP tokenizer revision
  `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- GGML commit `331b9cba52b23d895bc4ad218c007eb5e667540f`

## Output quality and release motivation

We tested the released AnimateAnyMesh checkpoint with both the original
PyTorch implementation and this C++ implementation, including the official
dragon mesh and documented inference settings. Layer-by-layer and full
16-frame comparisons show that the C++ implementation closely reproduces the
original deformation. Despite that parity, we could not get either
implementation to produce consistently convincing animations. The documented
dragon sample is dominated by whole-mesh translation with comparatively little
purposeful deformation, while other prompt, seed, guidance, and trajectory
choices frequently produce weak motion or implausible warping.

This therefore appears to be a limitation of the released model and motion
prior rather than a C++ porting error. Numerical parity should not be read as a
claim of production-quality animation. The
[AnimateAnyMesh++ paper](https://arxiv.org/abs/2604.26917) describes changes
intended to improve motion diversity, trajectory sticking, geometry
preservation, and sequence length. Its code and weights are not public at the
time of this release, but we hope that model improves these results. We are
releasing this implementation as a transparent, tested baseline and as a
foundation for evaluating or porting AnimateAnyMesh++ when its weights become
available.

## Build

Nix provides the supported reproducible production environment. The release
build disables host-native compilation, builds every CPU ISA variant as a
loadable module, and builds Vulkan as another loadable module. GGML chooses the
best compatible CPU module at runtime.

```sh
nix build path:.
./result/bin/aam-cli backends
```

For development:

```sh
nix develop path:.
cmake --preset debug
cmake --build --preset debug -j
ctest --preset debug

cmake --preset release
cmake --build --preset release -j
./build/release/bin/aam-cli backends
```

The `debug` preset is a simple CPU build. The `release` preset enables Vulkan,
runtime backend loading, all CPU variants, stack protection, and fortified
libc calls. The public API is in `include/aam/aam.hpp`.

## Model conversion

Download the exact official inputs with atomic SHA-256 verification:

```sh
./scripts/download_models.sh checkpoints
mkdir -p models
docker build -t aam-reference:2.6 reference
docker run --rm \
  -v "$PWD:/work" \
  -v "$PWD/checkpoints:/checkpoints:ro" \
  aam-reference:2.6 /work/scripts/convert_to_gguf.py \
    --dvae /checkpoints/dvae_f.pth \
    --rf /checkpoints/rf_epoch_f.pth \
    --tokenizer /checkpoints/tokenizer \
    --output /work/models/animate-any-mesh-f16.gguf --ftype 1
```

`--ftype 1` stores matrix weights as F16 while retaining sensitive norms and
embeddings as F32; the resulting official model is about 456 MiB. Use
`--ftype 0` for the roughly 909 MiB all-F32 parity model. The converter refuses
unsafe pickle loading: it always uses `torch.load(weights_only=True, mmap=True)`
and writes the GGUF atomically.

Expected source hashes are embedded in the downloader and GGUF metadata:

| Artifact | SHA-256 |
|---|---|
| `dvae_f.pth` | `af78499ca80c8761086cbc6cd48e6dcc735e09238b64784621ac2b776db2c710` |
| `rf_epoch_f.pth` | `a5f1ad53e6616ef909f2234a0b80289c26538806cb397679c28279be4e122506` |
| `vocab.json` | `e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349` |
| `merges.txt` | `9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a` |

## Animate a mesh

```sh
./result/bin/aam-cli animate \
  --model models/animate-any-mesh-f16.gguf \
  --input examples/dragon.glb \
  --output output/dragon-flying.glb \
  --prompt "The object is flying" \
  --device vulkan --seed 666 --steps 64 --guidance 3.0 --fps 10
```

Use `--device cpu` to require CPU or `--device auto` to prefer Vulkan and fall
back to CPU. `--backend-dir` overrides the module search directory. Other useful
options are `--threads`, `--trajectories`, `--max-vertices`, and `--fps`.
The default trajectory count matches upstream: `max(512, vertices / 8)`.
`--max-vertices` is the inference-topology budget, not an upload limit. Dense
inputs are deterministically simplified to that budget before model loading;
the exported animation uses the simplified topology. Inputs already within the
budget retain their original topology.
For steady-state profiling, `--warmups N --benchmark-runs N` loads the model
and preprocesses the mesh once, reports each synchronized generation to stderr,
and exports only the last result. The public `prepared_mesh` overload provides
the same reuse to embedding applications.

Recommended generation settings:

| Setting | Meaning | Start with |
|---|---|---|
| Prompt | Motion caption used for conditioning | A short literal phrase. The official dragon uses `The object is flying`. |
| Seed | Sampling variation; it can change motion substantially | `666` reproduces the pinned upstream dragon sample in strict parity mode. |
| Steps | Midpoint solver evaluations; this affects refinement, not duration | `64`; use `32` for previews. |
| Guidance | Strength of the prompt relative to the unconditional sample | `3`; try `2` for distortion or `4` if motion is ignored. |
| FPS | GLB playback rate only; inference always produces 16 frames | `10`, matching upstream and giving a 1.5-second keyframe span. |
| Trajectories | Sampled motion anchors | `0` for automatic. Fewer amplify motion but weaken shape preservation. |
| Inference vertices | Deterministic simplification budget | `10000`; lower is faster, while higher retains more geometric detail. |

The default strict parity path reproduces PyTorch 2.7.1/CUDA 12.8's Philox
stream after the pinned upstream RF architecture's model-construction offset.
It also computes the FPS-sensitive learned embedding/order on the dynamically
loaded CPU backend, because a numerically reordered trajectory set pairs the
same seeded RF noise with different mesh locations. `--fast-fps` skips that CPU
reference pass at the cost of seed parity. Like upstream, other seeds can still
vary sharply; increasing steps does not rescue an unsuitable sample.

The output is a standards-compliant GLB with 16 morph-target frames and a
linear weights animation. Input triangle primitives are flattened to world
space with scene-node matrix/TRS transforms and instancing applied. The output
is intentionally geometry-only: source materials, textures, cameras, lights,
skins, morph targets, and node hierarchy are not copied.

## Local gallery demo

The bundled browser app follows the local-first workflow of `trellis2cpp`'s
demo: upload GLB models into a gallery, select one, create any number of prompt
variations, and return to every saved animation after restarting the server.
It includes a dependency-free WebGL 2 viewer for both source meshes and the
exported morph-target animation, along with timeline controls and GLB download.

Build the C++ CLI and launch the separately packaged server:

```sh
nix build path:.
nix run path:.#demo -- \
  -model models/animate-any-mesh-f16.gguf \
  -device vulkan
```

Then open <http://127.0.0.1:8743>. The default durable store is `./demo-data`;
use `-store PATH` to put it elsewhere. Other server options include `-cli`,
`-addr`, `-threads`, `-timeout`, and `-max-upload-mib`. The server binds only to
localhost by default and sends uploads to one bounded inference worker, avoiding
concurrent jobs that would unexpectedly multiply GPU memory use. See
[`demo/README.md`](demo/README.md) for the storage layout, API, and security
notes.

## Layer-by-layer validation

The `reference/` image pins PyTorch, Transformers, NumPy, and the upstream
checkout interface. It emits deterministic GGUF fixtures for every major
boundary. See `reference/README.md` for container commands.

After producing `dumps/clip.gguf`, `dumps/vae.gguf`, and `dumps/rf.gguf`:

```sh
AAM_TEST_GGUF=models/animate-any-mesh-f32.gguf \
AAM_CLIP_REFERENCE=dumps/clip.gguf \
  ./build/debug/tests/aam-clip-parity

AAM_EXACT_ATTENTION=1 \
AAM_TEST_GGUF=models/animate-any-mesh-f32.gguf \
AAM_VAE_REFERENCE=dumps/vae.gguf \
  ./build/debug/tests/aam-vae-parity

AAM_EXACT_ATTENTION=1 \
AAM_TEST_GGUF=models/animate-any-mesh-f32.gguf \
AAM_RF_REFERENCE=dumps/rf.gguf \
  ./build/debug/tests/aam-rf-parity
```

`AAM_EXACT_ATTENTION=1` uses the unfused attention graph for strict numerical
comparison. Production defaults to GGML flash attention. On the pinned F32
model and CPU backend, validation measured:

| Boundary | Maximum absolute error | Relative L2 |
|---|---:|---:|
| CLIP final hidden state | 0.0020 | 2.26e-4 |
| CLIP residual blocks, worst | 0.030 | 5.1e-5 |
| VAE encoder | 6.0e-6 | below test threshold |
| VAE decoder internal, worst | 1.83e-4 | below test threshold |
| VAE final vertices | 1.19e-7 | below test threshold |
| RF final velocity | 7.27e-6 | 1.2e-6 |
| Midpoint sampler fixture | 1.63e-5 | below test threshold |

Vulkan layer validation additionally caught a backend-specific point-embedding
failure: GLSL `sin`/`cos` lost significant accuracy at the VAE's high Fourier
frequencies. The production path now computes that small deterministic feature
map on the host. On the full official dragon fixture this reduced the Vulkan
VAE encoded relative L2 error from 1.08e-2 to 1.24e-3; the small fixture's
pre-neighbour error fell from 2.28e-2 to 4.28e-4.

The full 16-frame deformation comparison then exposed a discontinuous error
that ordinary layer norms concealed. Given PyTorch's exact trajectory indices,
epsilon, and RF noise, Vulkan's frame displacement direction cosine is
0.999934 or better and its worst relative displacement error is 1.30%. When
production recomputed FPS from Vulkan embeddings, 990/1,020 selected vertices
overlapped upstream but only 428 retained the same order; pairing noise by that
new order changed frame motion by 36--55%. The hybrid model therefore retains
the point/neighbor embedding matrices in F32 and strict mode computes that
small learned embedding on CPU, restoring all 1,020 trajectory indices in
order before running RF and decoding on Vulkan.

The tokenizer tests also cover accented Latin text, mixed-width digits,
Chinese, emoji, Roman numerals, HTML entities, Turkish dotted I, and German
sharp S against the pinned Transformers tokenizer.

## Safety and fuzzing

The library uses owning containers, spans, `std::expected`, checked size
arithmetic, finite-value checks, explicit input limits, strict GLB accessor
bounds, cycle detection in node graphs, stable seeded random generation, and
atomic model/output publication. GGUF files are treated as converted model
artifacts; obtain them through the verified pipeline above.

Run ASan and UBSan over project code:

```sh
nix develop path:.#fuzz
cmake --preset asan
cmake --build --preset asan -j
ctest --preset asan
```

Run the libFuzzer harnesses for GLB parsing, mesh preparation/simplification,
and farthest point sampling:

```sh
cmake --preset fuzz
cmake --build --preset fuzz -j
mkdir -p corpus/glb corpus/mesh corpus/fps
./build/fuzz/fuzz/fuzz_glb corpus/glb -max_total_time=300
./build/fuzz/fuzz/fuzz_mesh corpus/mesh -max_total_time=300
./build/fuzz/fuzz/fuzz_fps corpus/fps -max_total_time=300
```

Default parser limits are 256 MiB per GLB, 16 MiB of JSON, 1,048,576 input
vertices, and 2,097,152 faces. Generation defaults to 4,096 merged vertices and
hard-limits trajectories to 16,384. The gallery caps its inference budget at
16,384 vertices. Increase the CLI's `--max-vertices` deliberately; model memory
and work grow substantially with the vertex count.

## Vulkan performance

The production path uses F16 matrix weights while retaining the small
FPS-order-sensitive VAE embedding matrices in F32, GGML flash attention, batched
conditional/unconditional inference with CFG fused onto the device graph, an
exact deterministic FPS implementation with a persistent CPU worker team,
strided RF views that avoid materializing tensor slices, a persistent graph
allocator across all rectified-flow evaluations, and a synchronized two-stream
VAE attention graph that shares score/softmax work. CPU modules remain
dynamically linked and are compiled for the full GGML ISA matrix instead of the
build host.

For a defensible comparison, benchmark the same physical GPU, mesh, prompt,
seed, trajectory count, 64 sampling steps, guidance, and precision. Report
steady inference (load and preprocess once, then one warm-up and at least five
synchronized runs) separately from fresh-process latency (at least five
processes including model loading and export). Report median wall time and peak
device memory for each scope. Compare this CLI against upstream `test_drive.py`
with rendering and FBX/ABC export disabled.

On NixOS, the packaged CLI and development shell include both the pinned Vulkan
loader and `/run/opengl-driver/lib`, so the proprietary NVIDIA ICD is discovered
without manual environment setup. GGML detects the development RTX 5070 Ti with
FP16, BF16, and `NV_coopmat2` support. Reproduce measured results with
`scripts/benchmark_cpp.sh`; benchmark numbers are hardware- and driver-specific
and should always include the exact command and model precision.

On 2026-08-11, an RTX 5070 Ti with NVIDIA driver 595.71.05 produced the following
steady-state results for the official dragon (9,639 source / 8,160 merged
vertices, 16,074 faces), prompt `The object is flying`, 1,020 automatic
trajectories, 64 midpoint steps, and guidance 3.0. Model loading, mesh
preprocessing, rendering, and export are outside the timed region for both
implementations. Each row uses one warm-up and five synchronized runs; the
PyTorch rows use the exact pure-Torch replacement for unavailable PyTorch3D
farthest-point sampling.

| Implementation | Weight precision | Median | Throughput vs eager |
|---|---:|---:|---:|
| Upstream PyTorch eager | F32 | 4.10775 s | 1.00x |
| Upstream PyTorch `torch.compile` | F32 | 3.40833 s | 1.21x |
| C++ / GGML Vulkan, strict FPS parity | Hybrid F16/F32 | 1.91767 s | **2.14x** |

The strict production path is therefore 2.14x as fast as upstream eager and
1.78x as fast as upstream `torch.compile`. It spends about 0.23--0.29 seconds
computing the learned FPS embedding on CPU so that seeded noise remains paired
with the same trajectories as upstream. `--fast-fps` omits that pass, but is a
throughput/debug option rather than the validated output path. The compile row wraps the unchanged
VAE and RF modules with `torch.compile`; it works without model surgery, but its
first warm-up took 30.47 seconds and Dynamo reports graph breaks in tokenizer
Unicode handling and the exact FPS loop. Profiling found that FPS dominated the
original C++ steady run; parallelizing only the independent candidate scan
reduced it to roughly 0.41--0.62 seconds here while preserving the ordered
reduction exactly.
Strided AdaLN, residual, and QKV views then reduced GGML `CONT` dispatches from
33,332 to 320 and Vulkan timestamped operation time from 1.656 s to 1.496 s.
Set `AAM_PROFILE=1` to emit per-stage and aggregate graph-execution timings.

Upstream used the unchanged model code and checkpoints on PyTorch 2.7.1+cu128
because its pinned PyTorch 2.6+cu124 wheel has no `sm_120` kernels. The
unavailable PyTorch3D FPS operation was replaced with an exact pure-Torch
implementation. Separate pre-optimization 100 ms
`nvidia-smi` sampling measured approximate peak GPU-memory deltas of 2,252 MiB
for C++ F16 and 1,830 MiB for upstream F32.

## License

Apache-2.0. This is an independent C++ reimplementation derived from the
published Apache-2.0 AnimateAnyMesh code and model architecture. See `NOTICE`
for attribution.
