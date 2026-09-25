"""Train Vane text on a Modal A100 40GB.

First run measures a few steps and saves a checkpoint. It does not start the
full epoch. $30 at the published A100-40GB rate is about 14 hours.

    modal run modal_train_text.py
    modal run modal_train_text.py --steps 20
"""

from __future__ import annotations

import modal

app = modal.App("vane-text")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.48",
        "datasets>=3.0",
        "pyyaml>=6.0",
        "pillow>=10.0",
        "huggingface_hub>=0.26",
        "safetensors>=0.4",
        "numpy>=1.26",
    )
    .add_local_dir("src/vane", remote_path="/root/vane")
)

volume = modal.Volume.from_name("vane-text-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60,
    volumes={"/checkpoints": volume},
)
def train(steps: int = 20, batch_size: int = 2) -> dict:
    import os
    import sys
    import time

    sys.path.insert(0, "/root")
    os.environ.setdefault("HF_HOME", "/checkpoints/hf")

    from vane.train.text import main as train_main

    config = "/root/vane/train/configs/vane-text-1.0.0.yaml"
    # The packaged tree lives at /root/vane because add_local_dir copies the package.
    # Rewrite a small overlay next to it so checkpoint_dir stays on the volume.
    overlay = "/tmp/vane-text-modal.yaml"
    text = open(config, encoding="utf-8").read()
    text = text.replace(
        "dir: checkpoints/vane-text-1.0.0",
        "dir: /checkpoints/vane-text-1.0.0",
    )
    text = text.replace("batch_size: 8", f"batch_size: {batch_size}")
    open(overlay, "w", encoding="utf-8").write(text)

    argv = ["vane-train", "--config", overlay, "--steps", str(steps)]
    sys.argv = argv
    started = time.perf_counter()
    train_main()
    elapsed = time.perf_counter() - started
    volume.commit()
    return {
        "steps": steps,
        "batch_size": batch_size,
        "seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / max(steps, 1), 2),
        "checkpoint_dir": "/checkpoints/vane-text-1.0.0",
    }


@app.local_entrypoint()
def main(steps: int = 20, batch_size: int = 2) -> None:
    result = train.remote(steps=steps, batch_size=batch_size)
    print(result)
