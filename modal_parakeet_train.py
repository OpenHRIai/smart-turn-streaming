"""Train the end-of-turn heads on the Parakeet encoder features, on Modal (the 16 GB cache does not fit a laptop).

    modal run modal_parakeet_train.py --name pk_ami07_auc --epochs 8
    modal volume get smart-turn-v2 runs/parakeet/pk_ami07_auc runs/parakeet/pk_ami07_auc

Runs ``scripts/b1_train.py --backbone parakeet`` unchanged otherwise. The default arguments are the recipe of
the released smart-turn-streaming heads (B1 run ``knob_ami07_auc``: lr 1e-3, 8 epochs, checkpoint by dev AUC,
AMI weight 0.7, dropout 0.1), so the two heads differ only in the encoder they read.
"""

from __future__ import annotations

import modal

app = modal.App("parakeet-eou-train")
volume = modal.Volume.from_name("smart-turn-v2")
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.15.0", "numpy==2.3.4", "pandas", "pyarrow", "scikit-learn", "matplotlib", "tabulate")
    .add_local_python_source("eot")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


@app.function(image=image, gpu="L4", cpu=4.0, memory=49152, volumes={VOL: volume}, timeout=4 * 3600)
def train(name: str, args: list[str]) -> str:
    import subprocess
    out = f"{VOL}/runs/parakeet/{name}"
    cmd = ["python", "/root/scripts/b1_train.py", "--backbone", "parakeet", "--feats", f"{VOL}/data/parakeet/feats.npz",
           "--items", f"{VOL}/data/b1/items.parquet", "--out", out, *args]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root")
    volume.commit()
    return out


@app.local_entrypoint()
def main(name: str = "pk_ami07_auc", epochs: int = 8, lr: float = 1e-3, select: str = "auc", ami_weight: float = 0.7,
         dropout: float = 0.1, extra: str = ""):
    args = ["--epochs", str(epochs), "--lr", str(lr), "--select", select, "--ami-weight", str(ami_weight),
            "--dropout", str(dropout), *extra.split()]
    print(train.remote(name, args))
