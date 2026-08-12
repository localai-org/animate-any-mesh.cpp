#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-checkpoints}"
model_revision="d3360418409cc51d7bdc0c9633dfb93844113696"
tokenizer_revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

mkdir -p "$output_dir/tokenizer"

fetch() {
  local url="$1"
  local destination="$2"
  local expected="$3"
  local partial="${destination}.partial"
  trap 'rm -f -- "$partial"' RETURN
  curl --fail --location --retry 3 --output "$partial" "$url"
  printf '%s  %s\n' "$expected" "$partial" | sha256sum --check --status -
  mv -- "$partial" "$destination"
  trap - RETURN
}

fetch "https://huggingface.co/JarrentWu/AnimateAnyMesh/resolve/$model_revision/dvae/dvae_f.pth" \
  "$output_dir/dvae_f.pth" af78499ca80c8761086cbc6cd48e6dcc735e09238b64784621ac2b776db2c710
fetch "https://huggingface.co/JarrentWu/AnimateAnyMesh/resolve/$model_revision/rf_model/rf_epoch_f.pth" \
  "$output_dir/rf_epoch_f.pth" a5f1ad53e6616ef909f2234a0b80289c26538806cb397679c28279be4e122506
fetch "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/$tokenizer_revision/tokenizer/vocab.json" \
  "$output_dir/tokenizer/vocab.json" e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349
fetch "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/$tokenizer_revision/tokenizer/merges.txt" \
  "$output_dir/tokenizer/merges.txt" 9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a

echo "verified pinned AnimateAnyMesh checkpoints and CLIP tokenizer in $output_dir"
