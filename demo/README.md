# Local gallery demo

This is a small, local-first web app inspired by the `trellis2cpp` demo. It
keeps uploaded GLB models in a gallery and stores multiple generated animations
under each model. The frontend, LocalAI mark, CSS, and JavaScript are embedded
in one Go binary; it has no runtime web or CDN dependencies.

## Run

From the repository root:

```sh
nix build path:.
nix run path:.#demo -- \
  -cli ./result/bin/aam-cli \
  -model models/animate-any-mesh-f16.gguf \
  -device vulkan \
  -store ./demo-data
```

Open <http://127.0.0.1:8743>. Use `-device cpu` to require the dynamically
loaded GGML CPU backend or `-device auto` to prefer Vulkan and fall back to CPU.
The default address is loopback-only; exposing the app to a network requires an
explicit `-addr` and an authentication-aware reverse proxy.

The server runs one animation at a time through `aam-cli`. Arguments are passed
as an argument vector, never through a shell. Each job has a configurable
timeout (30 minutes by default), subprocess output is bounded, uploads are
size-limited and GLB-header checked, IDs are cryptographically random, and
state is published with temporary-directory renames.

Dense uploads are automatically simplified by the CLI to the selected
inference-vertex budget (10,000 by default, capped at 16,384 in the gallery).
The original upload remains in the gallery; generated GLBs use the bounded
simplified topology.

## Persistence

The store contains one directory per uploaded model:

```text
demo-data/
  MODEL_ID/
    manifest.json
    source.glb
    animations/
      ANIMATION_ID/
        manifest.json
        animation.glb
```

Model thumbnails are compact data URLs in the model manifest. Interrupted
uploads and generations remain hidden temporary directories and are ignored on
restore. Deleting a model removes its complete gallery only after the server
has verified the resolved model path and confirmed that no job is active.

## API

- `GET, POST /api/models` lists models or accepts multipart field `model`.
- `GET, DELETE /api/model/{id}` reads or deletes a gallery model.
- `GET /api/model/{id}/source` serves the uploaded GLB.
- `POST /api/model/{id}/animations` queues an animation form.
- `GET /api/job/{id}` polls queued, running, done, or error state.
- `GET, DELETE /api/animation/{id}` serves or deletes a saved animation.

Run the server tests, including concurrent job polling under the race detector:

```sh
nix develop path:.
cd demo
go test -race ./...
go vet ./...
go test -fuzz=FuzzValidateUploadedGLB -fuzztime=30s
```
