"""
Hybrid Two-Stage Training Pipeline for Chimera → Wyrd

Stage 1: Eloquent Prose
    - Train on Gutenberg classics (Wilde, Lovecraft, Dunsany, etc.)
    - Goal: Establish sophisticated linguistic register
    - Replace TinyStories "See Spot Run" syntax

Stage 2: Quality Narrative
    - Fine-tune on narrative fiction with good story structure
    - Goal: Recover storytelling ability without losing prose quality
    - Think Le Guin, Gaiman, Bradbury style (but public domain equivalents)

Usage:
    # Full pipeline
    python train_hybrid.py --model-path checkpoints/model_1500.pt --full-pipeline

    # Just Stage 1
    python train_hybrid.py --model-path checkpoints/model_1500.pt --stage 1

    # Just Stage 2 (after Stage 1)
    python train_hybrid.py --model-path checkpoints/prose_stage1_final.pt --stage 2
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(1)


def download_data(stage: int):
    """Download prose data from Gutenberg."""
    print(f"\n Downloading Stage {stage} data...")

    cmd = [
        sys.executable, "-u", "download_prose.py",
        "--stage", str(stage),
        "--merge",
        "--output-dir", "data/prose",
    ]

    run_command(cmd, f"Download Stage {stage} Data")


def train_stage(
    stage: int,
    model_path: str,
    model_config: str,
    lr: float,
    epochs: int,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    seq_length: int,
    cpu_offload: bool,
):
    """Run training for a stage."""
    if stage == 1:
        data_file = "data/prose/eloquent_merged.txt"
        output_prefix = "prose_stage1"
        stage_name = "Eloquent Prose"
    else:
        data_file = "data/prose/narrative_merged.txt"
        output_prefix = "prose_stage2"
        stage_name = "Quality Narrative"

    # Check data exists
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        print("Run with --download-data to fetch it first")
        sys.exit(1)

    print(f"\n Training Stage {stage}: {stage_name}")

    cmd = [
        sys.executable, "-u", "train_prose.py",
        "--model-path", model_path,
        "--model-config", model_config,
        "--data-path", data_file,
        "--seq-length", str(seq_length),
        "--stride", str(seq_length // 2),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(grad_accum),
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--output-dir", "checkpoints",
        "--save-every", "500",
    ]

    if max_steps > 0:
        cmd.extend(["--max-steps", str(max_steps)])

    if cpu_offload:
        cmd.append("--cpu-offload")

    run_command(cmd, f"Stage {stage}: {stage_name}")

    # Rename final checkpoint
    final_src = Path("checkpoints/prose_final.pt")
    final_dst = Path(f"checkpoints/{output_prefix}_final.pt")
    if final_src.exists():
        final_src.rename(final_dst)
        print(f"Renamed: {final_src} -> {final_dst}")

    return str(final_dst)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Two-Stage Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download data + full pipeline
    python train_hybrid.py --model-path checkpoints/model_1500.pt --download-data --full-pipeline

    # Just download data
    python train_hybrid.py --download-data

    # Run only Stage 1
    python train_hybrid.py --model-path checkpoints/model_1500.pt --stage 1

    # Run Stage 2 on Stage 1 output
    python train_hybrid.py --model-path checkpoints/prose_stage1_final.pt --stage 2

    # Memory-constrained (6GB VRAM): use CPU offload
    python train_hybrid.py --model-path checkpoints/model_1500.pt --full-pipeline --cpu-offload

    # Quick test run (100 steps per stage)
    python train_hybrid.py --model-path checkpoints/model_1500.pt --full-pipeline --max-steps 100
"""
    )

    # Model
    parser.add_argument("--model-path", type=str, default="checkpoints/model.pt",
                        help="Path to starting model weights")
    parser.add_argument("--model-config", type=str, default="medium",
                        choices=["small", "medium", "base"])

    # Pipeline control
    parser.add_argument("--download-data", action="store_true",
                        help="Download prose data from Gutenberg")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run both Stage 1 and Stage 2")
    parser.add_argument("--stage", type=int, choices=[1, 2],
                        help="Run specific stage only")

    # Training hyperparams
    parser.add_argument("--lr1", type=float, default=1e-5,
                        help="Learning rate for Stage 1")
    parser.add_argument("--lr2", type=float, default=5e-6,
                        help="Learning rate for Stage 2 (lower)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max steps per stage (-1 = full epochs)")

    # Memory optimization
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload optimizer to CPU (saves VRAM)")

    args = parser.parse_args()

    # Validate arguments
    if not args.download_data and not args.full_pipeline and args.stage is None:
        parser.print_help()
        print("\nError: Must specify --download-data, --full-pipeline, or --stage")
        sys.exit(1)

    # Download data if requested
    if args.download_data:
        download_data(1)
        download_data(2)
        if not args.full_pipeline and args.stage is None:
            print("\n Data downloaded. Run with --full-pipeline or --stage to train.")
            return

    # Training
    model_path = args.model_path

    if args.full_pipeline:
        # Stage 1
        model_path = train_stage(
            stage=1,
            model_path=model_path,
            model_config=args.model_config,
            lr=args.lr1,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seq_length=args.seq_length,
            cpu_offload=args.cpu_offload,
        )

        # Stage 2
        train_stage(
            stage=2,
            model_path=model_path,
            model_config=args.model_config,
            lr=args.lr2,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seq_length=args.seq_length,
            cpu_offload=args.cpu_offload,
        )

    elif args.stage == 1:
        train_stage(
            stage=1,
            model_path=model_path,
            model_config=args.model_config,
            lr=args.lr1,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seq_length=args.seq_length,
            cpu_offload=args.cpu_offload,
        )

    elif args.stage == 2:
        train_stage(
            stage=2,
            model_path=model_path,
            model_config=args.model_config,
            lr=args.lr2,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seq_length=args.seq_length,
            cpu_offload=args.cpu_offload,
        )

    print("\n" + "="*60)
    print("  HYBRID TRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Test generation:")
    print("     python generate.py --checkpoint checkpoints/prose_stage2_final.pt")
    print("")
    print("  2. Optional instruction fine-tuning:")
    print("     python train_instruct.py --model-path checkpoints/prose_stage2_final.pt")
    print("")
    print("  3. Chat with Wyrd:")
    print("     python chat_ui.py --checkpoint checkpoints/instruct_final.pt")


if __name__ == "__main__":
    main()
