"""
Self-Optimizing Chimera Framework

The complete system that:
1. Evolves architecture through mutation/crossover
2. Trains promising candidates efficiently
3. Evaluates on held-out data
4. Iterates to find optimal architecture for your hardware/data

Usage:
    python chimera_evolve.py --data-path data/tinystories.txt --budget 2h

This will run evolution within your time/compute budget and produce
the best architecture for your specific setup.
"""

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Chimera, ChimeraConfig
from tokenizer import ChimeraTokenizer
from train_packed import PackedDataset, Trainer, TrainConfig, create_optimizer, get_cosine_schedule
from evolution import (
    ArchitectureGenome, EvolutionarySearch, FitnessEvaluator,
    ScaleMutation, RecurrenceMutation, AttentionMutation,
    crossover
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for self-optimizing evolution."""
    # Data
    data_path: str = "data/tinystories.txt"
    val_split: float = 0.05  # 5% validation

    # Budget constraints
    max_time_hours: float = 2.0  # Total time budget
    max_params: int = 500_000_000  # 500M param limit

    # Evolution
    population_size: int = 6
    max_generations: int = 15
    elite_count: int = 2

    # Fitness evaluation (short runs)
    eval_steps: int = 200
    eval_batch_size: int = 4
    eval_seq_length: int = 256

    # Full training (for winners)
    train_steps: int = 3000
    train_batch_size: int = 4
    train_seq_length: int = 512

    # Output
    output_dir: str = "chimera_evolved"


class SelfOptimizingChimera:
    """
    Main orchestrator for self-optimizing architecture search.

    Phases:
    1. EXPLORE: Quick evaluations to explore architecture space
    2. EXPLOIT: Longer training on promising candidates
    3. VALIDATE: Full evaluation on held-out data
    4. EXPORT: Save best model for deployment
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Tokenizer
        self.tokenizer = ChimeraTokenizer()
        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")

        # Timing
        self.start_time = None
        self.time_budget = config.max_time_hours * 3600  # seconds

        # State
        self.generation = 0
        self.best_genome: Optional[ArchitectureGenome] = None
        self.best_val_loss = float('inf')
        self.history: List[Dict] = []

    def time_remaining(self) -> float:
        """Get remaining time budget in seconds."""
        if self.start_time is None:
            return self.time_budget
        return max(0, self.time_budget - (time.time() - self.start_time))

    def should_continue(self) -> bool:
        """Check if we should continue evolution."""
        return self.time_remaining() > 0 and self.generation < self.config.max_generations

    def quick_evaluate(self, genome: ArchitectureGenome) -> Dict[str, float]:
        """Quick fitness evaluation (short training run)."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        try:
            # Build model
            model_config = genome.to_config(vocab_size=self.tokenizer.vocab_size)
            model = Chimera(model_config).to(self.device)

            # Check param count
            params = sum(p.numel() for p in model.parameters())
            if params > self.config.max_params:
                logger.warning(f"Model too large: {params:,} > {self.config.max_params:,}")
                return {"loss": float('inf'), "params": params}

            # Dataset
            dataset = PackedDataset(
                self.config.data_path,
                self.tokenizer,
                self.config.eval_seq_length,
            )
            loader = DataLoader(dataset, batch_size=self.config.eval_batch_size)

            # Quick training
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            model.train()

            total_loss = 0.0
            data_iter = iter(loader)
            start = time.time()

            for step in range(self.config.eval_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                    logits, _ = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()

            elapsed = time.time() - start
            avg_loss = total_loss / self.config.eval_steps
            toks_per_sec = (self.config.eval_steps * self.config.eval_batch_size *
                          self.config.eval_seq_length) / elapsed

            return {
                "loss": avg_loss,
                "ppl": math.exp(min(avg_loss, 20)),
                "params": params,
                "toks_per_sec": toks_per_sec,
                "time": elapsed,
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"loss": float('inf'), "params": 0, "error": str(e)}

        finally:
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def full_train(self, genome: ArchitectureGenome) -> Dict[str, float]:
        """Full training run for promising candidate."""
        logger.info(f"Starting full training for genome {genome.get_hash()}")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        try:
            # Build model
            model_config = genome.to_config(vocab_size=self.tokenizer.vocab_size)
            model = Chimera(model_config).to(self.device)

            # Dataset
            train_dataset = PackedDataset(
                self.config.data_path,
                self.tokenizer,
                self.config.train_seq_length,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.train_batch_size
            )

            # Training config
            train_config = TrainConfig(
                micro_batch_size=self.config.train_batch_size,
                gradient_accumulation_steps=4,
                learning_rate=3e-4,
                max_steps=self.config.train_steps,
                warmup_steps=min(200, self.config.train_steps // 10),
                seq_length=self.config.train_seq_length,
            )

            # Trainer
            trainer = Trainer(model, train_config, self.device)

            # Training loop
            model.train()
            data_iter = iter(train_loader)
            start = time.time()
            micro_step = 0

            while trainer.step < self.config.train_steps:
                # Time check
                if self.time_remaining() < 60:  # Save 1 min buffer
                    logger.warning("Time budget exhausted during training")
                    break

                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                trainer.train_step(batch)
                micro_step += 1

                if micro_step % train_config.gradient_accumulation_steps == 0:
                    trainer.optimizer_step()

                    if trainer.step % 100 == 0:
                        metrics = trainer.get_metrics()
                        logger.info(f"  Step {trainer.step}: loss={metrics['loss']:.4f}")

            elapsed = time.time() - start
            final_loss = trainer.get_metrics()["loss"]

            # Save checkpoint
            checkpoint_path = self.output_dir / f"trained_{genome.get_hash()}.pt"
            torch.save({
                "model": model.state_dict(),
                "genome": asdict(genome),
                "loss": final_loss,
            }, checkpoint_path)

            return {
                "loss": final_loss,
                "ppl": math.exp(min(final_loss, 20)),
                "steps": trainer.step,
                "time": elapsed,
                "checkpoint": str(checkpoint_path),
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"loss": float('inf'), "error": str(e)}

        finally:
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def evolve_generation(self, population: List[ArchitectureGenome]) -> List[ArchitectureGenome]:
        """Run one generation of evolution."""
        logger.info(f"\n{'='*60}")
        logger.info(f"GENERATION {self.generation + 1}")
        logger.info(f"Time remaining: {self.time_remaining()/60:.1f} min")
        logger.info(f"{'='*60}")

        # Evaluate all genomes
        results = []
        for i, genome in enumerate(population):
            logger.info(f"\nEvaluating {i+1}/{len(population)}: {genome.get_hash()}")
            logger.info(f"  d_model={genome.d_model}, layers={genome.n_layers}, "
                       f"rec_ratio={genome.recurrence_ratio}")

            metrics = self.quick_evaluate(genome)
            results.append((genome, metrics))
            logger.info(f"  Loss: {metrics['loss']:.4f}, Params: {metrics.get('params', 0):,}")

            if not self.should_continue():
                break

        # Sort by loss (lower is better)
        results.sort(key=lambda x: x[1]["loss"])

        # Record best
        best_genome, best_metrics = results[0]
        self.history.append({
            "generation": self.generation,
            "best_loss": best_metrics["loss"],
            "best_hash": best_genome.get_hash(),
            "best_params": best_metrics.get("params", 0),
            "population_size": len(population),
            "time_remaining": self.time_remaining(),
        })

        # Check if new best
        if best_metrics["loss"] < self.best_val_loss:
            self.best_val_loss = best_metrics["loss"]
            self.best_genome = best_genome
            logger.info(f"\n*** NEW BEST: {best_genome.get_hash()} loss={best_metrics['loss']:.4f} ***")

        # Create next generation
        # Keep elites
        elites = [g for g, _ in results[:self.config.elite_count]]

        # Create offspring
        mutators = [ScaleMutation(), RecurrenceMutation(), AttentionMutation()]
        offspring = []

        parents = [g for g, m in results if m["loss"] < float('inf')][:4]
        if not parents:
            parents = elites

        while len(offspring) < self.config.population_size - len(elites):
            # Select parent
            parent = parents[len(offspring) % len(parents)]

            # Crossover (30% chance)
            if len(parents) > 1 and torch.rand(1).item() < 0.3:
                other = parents[(len(offspring) + 1) % len(parents)]
                child = crossover(parent, other)
            else:
                child = ArchitectureGenome(**asdict(parent))

            # Mutate
            mutator = mutators[len(offspring) % len(mutators)]
            child = mutator.mutate(child)

            offspring.append(child)

        next_generation = elites + offspring
        self.generation += 1

        return next_generation

    def run(self) -> ArchitectureGenome:
        """Run the complete self-optimization process."""
        logger.info("=" * 60)
        logger.info("SELF-OPTIMIZING CHIMERA")
        logger.info(f"Budget: {self.config.max_time_hours}h")
        logger.info(f"Max params: {self.config.max_params:,}")
        logger.info("=" * 60)

        self.start_time = time.time()

        # Initialize population with diverse seeds
        population = [
            ArchitectureGenome(),  # Default
            ArchitectureGenome(d_model=512, n_layers=8, n_heads=8, n_kv_heads=2),
            ArchitectureGenome(d_model=768, n_layers=12, n_heads=12, recurrence_ratio=4),
            ArchitectureGenome(d_model=640, n_layers=10, n_heads=8, window_size=256),
            ArchitectureGenome(d_model=896, n_layers=8, n_heads=8, ffn_multiplier=3.0),
            ArchitectureGenome(d_model=768, n_layers=16, n_heads=12, recurrence_ratio=2),
        ]

        # Evolution loop
        while self.should_continue():
            population = self.evolve_generation(population)

            # Save checkpoint
            self.save_state()

        # PHASE 2: Full training on best candidate
        if self.best_genome and self.time_remaining() > 300:  # 5 min buffer
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 2: FULL TRAINING")
            logger.info("=" * 60)

            train_results = self.full_train(self.best_genome)
            logger.info(f"Final training loss: {train_results.get('loss', 'N/A')}")

        # Export results
        self.export_results()

        logger.info("\n" + "=" * 60)
        logger.info("EVOLUTION COMPLETE")
        logger.info(f"Best genome: {self.best_genome.get_hash() if self.best_genome else 'None'}")
        logger.info(f"Best loss: {self.best_val_loss:.4f}")
        logger.info(f"Total time: {(time.time() - self.start_time)/60:.1f} min")
        logger.info("=" * 60)

        return self.best_genome

    def save_state(self):
        """Save current evolution state."""
        state = {
            "generation": self.generation,
            "best_val_loss": self.best_val_loss,
            "best_genome": asdict(self.best_genome) if self.best_genome else None,
            "history": self.history,
            "config": asdict(self.config),
        }
        with open(self.output_dir / "evolution_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def export_results(self):
        """Export final results."""
        if self.best_genome is None:
            logger.warning("No best genome found")
            return

        # Save best genome
        with open(self.output_dir / "best_genome.json", "w") as f:
            json.dump(asdict(self.best_genome), f, indent=2)

        # Save as ChimeraConfig
        config = self.best_genome.to_config(vocab_size=self.tokenizer.vocab_size)
        with open(self.output_dir / "best_config.json", "w") as f:
            json.dump({
                "vocab_size": config.vocab_size,
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "n_kv_heads": config.n_kv_heads,
                "ffn_hidden": config.ffn_hidden,
                "max_seq_len": config.max_seq_len,
                "window_size": config.window_size,
                "recurrence_dim": config.recurrence_dim,
            }, f, indent=2)

        # Generate report
        report = f"""
# Chimera Evolution Report

## Best Architecture

- **Genome Hash**: {self.best_genome.get_hash()}
- **Validation Loss**: {self.best_val_loss:.4f}
- **Parameters**: ~{self.best_genome.estimate_params():,}

## Architecture Details

| Parameter | Value |
|-----------|-------|
| d_model | {self.best_genome.d_model} |
| n_layers | {self.best_genome.n_layers} |
| n_heads | {self.best_genome.n_heads} |
| n_kv_heads | {self.best_genome.n_kv_heads} |
| ffn_multiplier | {self.best_genome.ffn_multiplier} |
| recurrence_ratio | {self.best_genome.recurrence_ratio} |
| window_size | {self.best_genome.window_size} |

## Evolution History

Generations: {self.generation}
Time: {(time.time() - self.start_time)/60:.1f} minutes

## Usage

```python
from chimera import Chimera
from evolution import ArchitectureGenome
import json

# Load evolved genome
with open("{self.output_dir}/best_genome.json") as f:
    genome_dict = json.load(f)
genome = ArchitectureGenome(**genome_dict)

# Create model
config = genome.to_config(vocab_size=32000)
model = Chimera(config)
```
"""
        with open(self.output_dir / "EVOLUTION_REPORT.md", "w") as f:
            f.write(report)

        logger.info(f"Results exported to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Self-Optimizing Chimera")

    parser.add_argument("--data-path", type=str, default="data/tinystories.txt",
                        help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="chimera_evolved",
                        help="Output directory")
    parser.add_argument("--budget", type=str, default="1h",
                        help="Time budget (e.g., '2h', '30m')")
    parser.add_argument("--max-params", type=int, default=500_000_000,
                        help="Maximum parameters")
    parser.add_argument("--population-size", type=int, default=6,
                        help="Population size")
    parser.add_argument("--max-generations", type=int, default=15,
                        help="Maximum generations")

    args = parser.parse_args()

    # Parse time budget
    budget_str = args.budget.lower()
    if budget_str.endswith('h'):
        hours = float(budget_str[:-1])
    elif budget_str.endswith('m'):
        hours = float(budget_str[:-1]) / 60
    else:
        hours = float(budget_str)

    config = EvolutionConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_time_hours=hours,
        max_params=args.max_params,
        population_size=args.population_size,
        max_generations=args.max_generations,
    )

    optimizer = SelfOptimizingChimera(config)
    best = optimizer.run()

    if best:
        print(f"\nBest architecture found: {best.get_hash()}")
        print(f"Run full training with:")
        print(f"  python train_packed.py --data-path {args.data_path} --output-dir {args.output_dir}/trained")


if __name__ == "__main__":
    main()
