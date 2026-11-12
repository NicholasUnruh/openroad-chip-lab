"""GPT-2 small (124M) on FineWeb-Edu with every transformer-block Linear emulated bit-exactly.

Emulation scope: each Linear inside the transformer blocks (attention QKV and output projection,
MLP up and down projection) runs its forward, input-gradient and weight-gradient GEMMs through
kernel.gemm: block-scaled quantization along the GEMM's reduction axis, exact block dot products,
one FP32 rounding per block. The attention score and value products, embeddings, layer norms and
the tied LM head run in BF16 autocast, as in the BF16 baseline (the NVFP4 pretraining recipe,
arXiv 2509.25149, also keeps the embeddings and output head in higher precision).

Every format uses the same data order, seed and hyperparameters. The run checkpoints every
--checkpoint-minutes, exits with status 75 after --chunk-minutes or when another process is using
the GPU (the supervisor resumes it), and exits 0 once the token budget is done.

usage: uv run --extra cuda python train.py --name mxfp8 --format mxfp8 --data data --runs runs
"""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kernel
import mx

PAUSED = 75
FORMATS = {"bf16": None, **mx.FORMATS}

# Processes that are always on the GPU for the Windows desktop; anything else means someone is
# using the machine and the run yields.
DESKTOP = (
    "dwm.exe", "explorer.exe", "ShellHost.exe", "SearchHost.exe", "StartMenuExperienceHost.exe",
    "TextInputHost.exe", "LockApp.exe", "ApplicationFrameHost.exe", "SystemSettings.exe",
    "CrossDeviceResume.exe", "msedgewebview2.exe", "NVIDIA Overlay.exe", "PhoneExperienceHost.exe",
    "copilotapp.exe", "copilotapphost.exe", "[Insufficient Permissions]", "[Not Found]",
)  # fmt: skip


def foreign_gpu_process() -> str | None:
    """Name of a non-desktop process on the GPU, as the Windows driver reports it, if any."""
    cmd = ["/mnt/c/Windows/System32/nvidia-smi.exe", "--query-compute-apps=process_name",
           "--format=csv,noheader"]  # fmt: skip
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None  # not on the shared Windows machine
    for line in out.splitlines():
        name = line.strip().replace("\\", "/").rsplit("/", 1)[-1]
        if name and name not in DESKTOP:
            return name
    return None


class EmulatedLinear(torch.autograd.Function):
    """y = x W^T with all three GEMMs (forward, dX, dW) through the emulated datapath."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, x, weight, fmt):
        x2 = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(x2, weight)
        ctx.fmt = fmt
        return kernel.gemm(x2, weight, fmt).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dy):
        x2, weight = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).float()
        dx = kernel.gemm(dy2, weight.T, ctx.fmt)  # reduce over output features
        dw = kernel.gemm(dy2.T, x2.T, ctx.fmt)  # reduce over tokens
        return dx.reshape(*dy.shape[:-1], x2.shape[-1]), dw, None


class Linear(nn.Linear):
    def __init__(self, n_in: int, n_out: int, fmt: mx.Format | None):
        super().__init__(n_in, n_out, bias=False)
        self.fmt = fmt

    def forward(self, x):
        if self.fmt is None:
            return super().forward(x)
        return EmulatedLinear.apply(x, self.weight, self.fmt)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, fmt: mx.Format | None):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = Linear(d, 3 * d, fmt), Linear(d, d, fmt)
        self.fc, self.down = Linear(d, 4 * d, fmt), Linear(4 * d, d, fmt)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, fmt: mx.Format | None, layers=12, d=768, heads=12, vocab=50304, ctx=1024):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, fmt) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight  # tied
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = (
                    0.02 / math.sqrt(2 * layers)
                    if name.endswith(("proj.weight", "down.weight"))
                    else 0.02
                )
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx, targets):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))


class Batches:
    """Deterministic data order: a seeded permutation of non-overlapping windows, by step."""

    def __init__(self, path: Path, ctx: int, seed: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx = ctx
        windows = (len(self.tokens) - 1) // ctx
        self.order = np.random.default_rng(seed).permutation(windows)

    def get(self, index: int, count: int, device="cuda"):
        rows = [self.order[(index + i) % len(self.order)] * self.ctx for i in range(count)]
        x = np.stack([self.tokens[r : r + self.ctx + 1] for r in rows]).astype(np.int64)
        x = torch.from_numpy(x).to(device, non_blocking=True)
        return x[:, :-1], x[:, 1:]


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, val: Batches, batches: int, micro: int) -> float:
    model.eval()
    total = 0.0
    for i in range(batches):
        x, y = val.get(i * micro, micro)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total += float(model(x, y))
    model.train()
    return total / batches


def save(run: Path, step: int, model, opt, keep: int = 2) -> None:
    path = run / f"ckpt_{step:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()}, tmp)
    os.replace(tmp, path)
    for old in sorted(run.glob("ckpt_*.pt"))[:-keep]:
        old.unlink()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run directory name under --runs")
    p.add_argument("--format", choices=FORMATS, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--tokens", type=float, default=3e8, help="training token budget")
    p.add_argument("--batch", type=int, default=64, help="sequences per optimizer step")
    p.add_argument("--micro", type=int, default=4, help="sequences per forward pass (8 GB)")
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--chunk-minutes", type=float, default=60)
    p.add_argument("--checkpoint-minutes", type=float, default=15)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (bench)")
    args = p.parse_args()

    run = args.runs / args.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "DONE").exists():
        return 0
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fmt = FORMATS[args.format]
    model = GPT(fmt).cuda()
    decay = [p for p in model.parameters() if p.dim() == 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True,
    )  # fmt: skip
    steps = int(args.tokens // (args.batch * args.ctx))
    warmup = max(1, steps // 50)
    step = 0
    ckpts = sorted(run.glob("ckpt_*.pt"))
    if ckpts:
        state = torch.load(ckpts[-1], map_location="cuda")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
    train = Batches(args.data / "train.bin", args.ctx, args.seed)
    val = Batches(args.data / "val.bin", args.ctx, args.seed + 1)
    log = open(run / "log.jsonl", "a")

    def record(**fields):
        log.write(json.dumps(fields) + "\n")
        log.flush()

    record(event="start", step=step, steps=steps, format=args.format, args=vars(args) | {
        "data": str(args.data), "runs": str(args.runs)})  # fmt: skip
    if steps * args.batch * args.ctx >= len(train.tokens):
        raise ValueError("token budget exceeds the training data")
    start = last_ckpt = time.time()
    last_check = 0.0
    accum = args.batch // args.micro
    status = PAUSED
    while step < steps:
        now = time.time()
        if now - last_check > 30:
            last_check = now
            other_proc = foreign_gpu_process()
            if other_proc:
                record(event="yield", step=step, process=other_proc)
                break
        if now - start > 60 * args.chunk_minutes or (args.max_steps and step >= args.max_steps):
            break
        if step % args.eval_every == 0:
            record(event="eval", step=step, tokens=step * args.batch * args.ctx,
                   val_loss=evaluate(model, val, args.eval_batches, args.micro))  # fmt: skip
        t0 = time.time()
        lr = lr_at(step, steps, args.lr, warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for micro in range(accum):
            x, y = train.get((step * accum + micro) * args.micro, args.micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y) / accum
            loss.backward()
            loss_sum += loss.item()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if kernel.range_errors is not None and int(kernel.range_errors):
            record(event="range_error", step=step)
            raise RuntimeError("block scales left the exact FP32 range")
        dt = time.time() - t0
        if not math.isfinite(loss_sum):
            record(event="diverged", step=step, loss=loss_sum)
            (run / "DONE").write_text("diverged\n")
            return 0
        record(event="train", step=step, loss=loss_sum, lr=lr, grad_norm=norm,
               tok_per_s=args.batch * args.ctx / dt)  # fmt: skip
        if time.time() - last_ckpt > 60 * args.checkpoint_minutes:
            save(run, step, model, opt)
            last_ckpt = time.time()
    if step >= steps:
        record(event="eval", step=step, tokens=step * args.batch * args.ctx,
               val_loss=evaluate(model, val, 4 * args.eval_batches, args.micro))  # fmt: skip
        status = 0
    if not args.max_steps:
        save(run, step, model, opt, keep=1 if status == 0 else 2)
    if status == 0:
        (run / "DONE").write_text("complete\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
