"""Download FineWeb-Edu sample shards and tokenize them with the GPT-2 BPE into uint16 files.

Writes <out>/train.bin and <out>/val.bin: documents joined by <|endoftext|>. The first
--val-docs documents of the first shard form the validation set. Each parquet shard is deleted
after tokenizing so disk use stays near the token files.

usage: uv run python prepare.py --out data --shards 1
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{:03d}_00000.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--val-docs", type=int, default=5000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    seen = 0
    with open(args.out / "train.bin", "wb") as train, open(args.out / "val.bin", "wb") as val:
        for shard in range(args.shards):
            path = args.out / f"shard_{shard:03d}.parquet"
            if not path.exists():
                print(f"downloading shard {shard}", flush=True)
                urllib.request.urlretrieve(URL.format(shard), path)
            table = pq.ParquetFile(path)
            for batch in table.iter_batches(batch_size=4096, columns=["text"]):
                docs = batch.column(0).to_pylist()
                for tokens in enc.encode_ordinary_batch(docs, num_threads=10):
                    out = val if seen < args.val_docs else train
                    out.write(np.array([eot, *tokens], dtype=np.uint16).tobytes())
                    seen += 1
            path.unlink()
            print(f"shard {shard}: {seen} documents, {train.tell() // 2} train tokens", flush=True)
    print(f"val tokens {(args.out / 'val.bin').stat().st_size // 2}")


if __name__ == "__main__":
    main()
