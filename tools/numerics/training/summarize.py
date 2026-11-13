"""Export finished GPT-2 runs into the journal: compact per-run logs and a results summary.

Reads <runs>/<name>/log.jsonl (train.py output) for every run with a DONE marker and writes
  docs/journal/data/training-runs/<name>/log.jsonl   eval and train records, one per step (a
                                                     resumed chunk repeats steps; the last wins)
  docs/journal/data/training-gpt2.json               final validation loss, the penalty against
                                                     BF16 at the same final step, throughput

usage: uv run --extra cpu python summarize.py --runs <runs dir>
"""

import argparse
import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "docs" / "journal" / "data"
KEEP = ("step", "loss", "tok_per_s", "exact_fallbacks")


def records(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().replace("\0", "").splitlines():  # abrupt stops leave NULs
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    args = p.parse_args()
    runs = {}
    for done in sorted(args.runs.glob("*/DONE")):
        name = done.parent.name
        rows = records(done.parent / "log.jsonl")
        evals = {r["step"]: r for r in rows if r["event"] == "eval"}
        train = {r["step"]: r for r in rows if r["event"] == "train"}
        shares = [r for r in rows if r["event"] == "share"]
        out = DATA / "training-runs" / name
        out.mkdir(parents=True, exist_ok=True)
        lines = [{"event": "eval", "step": s, "tokens": e["tokens"], "val_loss": e["val_loss"]}
                 for s, e in sorted(evals.items())]  # fmt: skip
        lines += [{"event": "train", **{k: t[k] for k in KEEP if k in t}} for _, t in sorted(train.items())]
        (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
        final = max(evals)
        rates = [t["tok_per_s"] for t in train.values()]
        runs[name] = {
            "status": done.read_text().strip(),
            "final_step": final,
            "final_tokens": evals[final]["tokens"],
            "final_val_loss": round(evals[final]["val_loss"], 4),
            "median_tokens_per_s": round(statistics.median(rates)),
            "steps_with_exact_fallback": sum(1 for t in train.values() if t.get("exact_fallbacks")),
            "exact_fallback_gemms": sum(t.get("exact_fallbacks", 0) for t in train.values()),
            "shared_gpu_events": len(shares),
        }
    base = runs.get("bf16")
    for name, r in runs.items():
        if base and name != "bf16" and r["final_step"] == base["final_step"]:
            r["penalty_pct_vs_bf16"] = round(100 * (r["final_val_loss"] / base["final_val_loss"] - 1), 2)
    summary = {
        "model": "GPT-2 124M, FineWeb-Edu, 300M tokens, seed 1337, identical data order",
        "scope": "emulated: transformer-block Linear GEMMs (forward, dgrad, wgrad); BF16: attention "
        "QK/AV, embeddings, tied LM head (arXiv 2509.25149)",
        "runs": runs,
    }
    (DATA / "training-gpt2.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
