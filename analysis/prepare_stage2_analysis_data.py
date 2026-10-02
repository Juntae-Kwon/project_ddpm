"""Create the compact, read-only Stage-2 analysis bundle used by the notebook."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "runs/two_stage_mnist_t100"
TARGET = Path(__file__).resolve().parent / "stage2_analysis_data"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    sources = {
        "experiment.json": SOURCE / "experiment.json",
        "schedule_summary.json": SOURCE / "schedule_summary.json",
        "stage1_completed.json": SOURCE / "stage1/completed.json",
        "stage2_config.json": SOURCE / "stage2/config.json",
        "pilot_indices.npy": SOURCE / "pilot_indices.npy",
        "posterior_samples.npz": SOURCE / "stage2/posterior_samples.npz",
        "diagnostics.npz": SOURCE / "stage2/diagnostics.npz",
        "stage1_ancestral_samples.npy": SOURCE / "stage1/final_ancestral_smoke.npy",
    }
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, TARGET / name)

    final_checkpoint = SOURCE / "stage2/checkpoints/chain_000400.npz"
    with np.load(final_checkpoint, allow_pickle=False) as checkpoint:
        if int(checkpoint["sweep"]) != 400:
            raise ValueError("the final checkpoint is not sweep 400")
        final_y_t = checkpoint["paths"][-1]
        if final_y_t.shape != (64, 28, 28, 1) or not np.isfinite(final_y_t).all():
            raise ValueError("invalid terminal latent state")
        np.save(TARGET / "final_y_T.npy", final_y_t)
        checkpoint_mu = checkpoint["mu"]

    records = [json.loads(line) for line in (SOURCE / "stage2/mcmc.jsonl").read_text().splitlines()]
    completion = next(record for record in reversed(records) if record.get("event") == "complete")
    with np.load(TARGET / "posterior_samples.npz", allow_pickle=False) as samples:
        np.testing.assert_array_equal(checkpoint_mu, samples["mu_all"][-1])
    checkpoint_metadata = {
        "source_checkpoint": str(final_checkpoint),
        "source_checkpoint_sha256": sha256(final_checkpoint),
        "sweep": 400,
        "latent_paths_shape": [100, 64, 28, 28, 1],
        "exported_terminal_shape": list(final_y_t.shape),
        "completion_record": completion,
    }
    (TARGET / "final_checkpoint_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2) + "\n"
    )

    manifest = []
    for path in sorted(TARGET.iterdir()):
        if path.name == "MANIFEST.txt":
            continue
        manifest.append(f"{path.name}\t{path.stat().st_size}\t{sha256(path)}")
    (TARGET / "MANIFEST.txt").write_text(
        "filename\tbytes\tsha256\n" + "\n".join(manifest) + "\n"
    )
    print(f"Prepared {len(manifest)} files ({sum(p.stat().st_size for p in TARGET.iterdir())} bytes): {TARGET}")


if __name__ == "__main__":
    main()
