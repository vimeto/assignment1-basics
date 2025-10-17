"""
Standalone evaluation script for debugging CUDA errors.
Loads a checkpoint and runs evaluation only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from cs336_basics.modules.cross_entropy import cross_entropy
from cs336_basics.modules.dataloader import dataloader
from cs336_basics.modules.transformer_lm import TransformerLM
from cs336_basics.modules.checkpointing import load_checkpoint


TORCH_PRECISIONS = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(device_str: str | None) -> torch.device:
    if device_str is None or device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_memmap_dataset(path: str) -> np.memmap:
    """Load a memory-mapped dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def evaluate(
    model: torch.nn.Module,
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    """Run evaluation on a dataset."""
    model.eval()
    losses: list[float] = []
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\nRunning evaluation:")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Context length: {context_length}")
        print(f"  Eval batches: {eval_batches}")
        print(f"  Dataset size: {len(dataset)}")

    with torch.no_grad():
        for batch_idx in range(eval_batches):
            try:
                X, Y = dataloader(
                    dataset=dataset,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device.type,
                    rng=rng,
                )

                if verbose and batch_idx == 0:
                    print(f"\nFirst batch info:")
                    print(f"  X shape: {X.shape}, device: {X.device}, dtype: {X.dtype}")
                    print(f"  Y shape: {Y.shape}, device: {Y.device}, dtype: {Y.dtype}")
                    print(f"  X range: [{X.min()}, {X.max()}]")

                # Check for issues before forward pass
                if torch.isnan(X).any() or torch.isinf(X).any():
                    raise ValueError(f"NaN/Inf in input X at batch {batch_idx}")

                logits = model(X)

                # Check for issues after forward pass
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError(f"NaN/Inf in logits at batch {batch_idx}")

                loss = cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    Y.reshape(-1),
                )

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    raise ValueError(f"NaN/Inf in loss at batch {batch_idx}")

                losses.append(float(loss.item()))

                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx + 1}/{eval_batches}: loss={losses[-1]:.4f}")

            except Exception as e:
                print(f"\nERROR at batch {batch_idx}/{eval_batches}:")
                print(f"  {type(e).__name__}: {e}")

                # Print debug info
                print(f"\nDebug info:")
                print(f"  X shape: {X.shape if 'X' in locals() else 'Not created'}")
                print(f"  Model device: {next(model.parameters()).device}")

                # Check model parameters
                print(f"\nChecking model parameters for NaN/Inf:")
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"    NaN in {name}")
                    if torch.isinf(param).any():
                        print(f"    Inf in {name}")

                raise

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    try:
        ppl = float(math.exp(mean_loss))
    except OverflowError:
        ppl = float("inf")

    return {"loss": mean_loss, "perplexity": ppl, "num_batches": len(losses)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a transformer model checkpoint")

    # Model config
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")

    # Data
    parser.add_argument("--val-path", type=str, required=True, help="Path to validation data")

    # Eval config
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])

    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra checks")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Setup
    device = resolve_device(args.device)
    dtype = TORCH_PRECISIONS.get(args.precision.lower())

    print("=" * 80)
    print("EVALUATION SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Validation data: {args.val_path}")
    print(f"  Device: {device}")
    print(f"  Precision: {args.precision} ({dtype})")
    print(f"  Debug mode: {args.debug}")

    if args.debug:
        print("\nEnabling debug checks...")
        torch.autograd.set_detect_anomaly(True)

    # Load validation data
    print(f"\nLoading validation dataset...")
    val_tokens = load_memmap_dataset(args.val_path)
    print(f"  Loaded {len(val_tokens)} tokens")

    # Build model
    print(f"\nBuilding model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    # Convert layers to ModuleList if needed
    if isinstance(model.layers, list):
        model.layers = torch.nn.ModuleList(model.layers)

    # Move model to device and dtype
    model = model.to(device=device, dtype=dtype)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    try:
        step = load_checkpoint(src=str(checkpoint_path), model=model, optimizer=None)
        print(f"  Loaded checkpoint from step {step}")
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")
        raise

    # Validate model parameters
    if args.debug:
        print(f"\nValidating model parameters...")
        has_issues = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"  WARNING: NaN in parameter {name}")
                has_issues = True
            if torch.isinf(param).any():
                print(f"  WARNING: Inf in parameter {name}")
                has_issues = True

        if not has_issues:
            print(f"  All parameters are valid (no NaN/Inf)")
        else:
            print(f"  WARNING: Model has corrupted parameters!")

    # Run evaluation
    print(f"\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80)

    try:
        metrics = evaluate(
            model=model,
            dataset=val_tokens,
            batch_size=args.batch_size,
            context_length=args.context_length,
            eval_batches=args.eval_batches,
            device=device,
            seed=args.seed,
            verbose=not args.quiet,
        )

        print(f"\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Batches evaluated: {metrics['num_batches']}")
        print(f"\nSUCCESS!")

    except Exception as e:
        print(f"\n" + "=" * 80)
        print("EVALUATION FAILED")
        print("=" * 80)
        print(f"  Error: {type(e).__name__}")
        print(f"  Message: {e}")
        print(f"\nFor more details, run with CUDA_LAUNCH_BLOCKING=1")
        raise


if __name__ == "__main__":
    main()
