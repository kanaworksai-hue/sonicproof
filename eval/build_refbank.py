"""Embed the reference clips and write src/sonicproof/data/refbank.npz (internal dev tool).

    uv run python eval/build_refbank.py [--audio-dir samples/refbank]

Only embeddings are shipped (no audio). They anchor a prompt's score: how much closer is the prompt to
this clip than to a broad bank of other sounds. Rebuild whenever the model changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sonicproof.audio import load_for_model
from sonicproof.model import MODEL_ID, Embedder

HERE = Path(__file__).parent
DEST = HERE.parent / "src" / "sonicproof" / "data" / "refbank.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(HERE / "refbank_v1.json"))
    ap.add_argument("--audio-dir", default="samples/refbank")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    titles = [c["title"] for c in spec["clips"]]
    emb = Embedder()
    R = emb.embed_audio([load_for_model(Path(args.audio_dir) / f"{t}.wav") for t in titles])
    # float16 is storage only; everything is computed in float32 after loading.
    np.savez_compressed(
        DEST, emb=R.astype(np.float16), titles=np.array(titles), model=MODEL_ID, name=spec["name"]
    )
    print(f"{DEST} {R.shape} {DEST.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
