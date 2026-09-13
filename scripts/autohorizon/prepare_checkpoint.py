"""Download and convert the public pi05 LIBERO checkpoint for native setup."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import gcsfs
from safetensors import safe_open


def download_tree(fs, remote, local):
    objects = fs.find(remote, detail=True)
    if not objects:
        raise RuntimeError(f"No checkpoint objects found at {remote}")
    for name, info in objects.items():
        if info.get("type") == "directory" or name.endswith("/"):
            continue
        relative = Path(name).relative_to(remote)
        destination = local / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == info["size"]:
            continue
        print(f"Downloading {name}", flush=True)
        partial = destination.with_name(destination.name + ".partial")
        fs.get_file(name, str(partial))
        if partial.stat().st_size != info["size"]:
            raise RuntimeError(f"Incomplete download: {name}")
        partial.replace(destination)


def check_weights(path):
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        # Detect conversion with unpatched Transformers, which silently drops
        # the pi05 adaptive normalization parameters in the upstream converter.
        for layer in range(18):
            for norm in ("input_layernorm", "post_attention_layernorm"):
                for suffix in ("weight", "bias"):
                    key = f"paligemma_with_expert.gemma_expert.model.layers.{layer}.{norm}.dense.{suffix}"
                    if key not in keys:
                        raise RuntimeError(f"Invalid pi05 checkpoint {path}: missing {key}")


def main():
    root = Path(os.environ["CHECKPOINT_ROOT"])
    fs = gcsfs.GCSFileSystem(token="anon")
    source = root / "pi05_libero"
    download_tree(fs, "openpi-assets/checkpoints/pi05_libero", source)

    # Pre-cache the tokenizer used by the policy's input transforms.
    tokenizer = root / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    remote = "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file() or tokenizer.stat().st_size != fs.info(remote)["size"]:
        fs.get_file(remote, str(tokenizer) + ".partial")
        Path(str(tokenizer) + ".partial").replace(tokenizer)

    target = root / "pi05_libero_pytorch_fixed"
    if target.exists():
        check_weights(target / "model.safetensors")
        shutil.copytree(source / "assets", target / "assets", dirs_exist_ok=True)
        print(f"Reusing verified checkpoint: {target}", flush=True)
        return

    staging = root / "pi05_libero_pytorch_fixed.partial"
    subprocess.run([
        sys.executable, "examples/convert_jax_model_to_pytorch.py",
        "--checkpoint_dir", str(source), "--config_name", "pi05_libero",
        "--output_path", str(staging),
    ], check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    check_weights(staging / "model.safetensors")
    shutil.copytree(source / "assets", staging / "assets", dirs_exist_ok=True)
    staging.rename(target)
    print(f"Converted checkpoint: {target}", flush=True)


if __name__ == "__main__":
    main()
