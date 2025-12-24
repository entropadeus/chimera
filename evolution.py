"""
Evolutionary Architecture Search for Chimera.

Implements:
- Genome representation of architecture choices
- Mutation operators for each component
- Crossover between successful architectures
- Fitness evaluation via training loss
- Population-based search with elitism

This turns Chimera into a self-optimizing architecture that discovers
optimal configurations for your hardware and data.
"""

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Chimera, ChimeraConfig
from tokenizer import ChimeraTokenizer
from train_packed import PackedDataset, create_optimizer, get_cosine_schedule


# ==============================================================================
# GENOME: Architecture as Evolvable DNA
# ==============================================================================

@dataclass
class ArchitectureGenome:
    """
    Genome encoding all evolvable architecture decisions.

    Each field represents a gene that can mutate.
    """
    # Core dimensions
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4  # For GQA

    # FFN
    ffn_multiplier: float = 2.67  # FFN hidden = d_model * multiplier
    ffn_activation: str = "swiglu"  # swiglu, gelu, relu

    # Recurrence vs Attention ratio (e.g., 3 = 3 recurrent per 1 attention)
    recurrence_ratio: int = 3

    # Sliding window
    window_size: int = 512

    # Recurrence settings
    recurrence_dim: int = 768  # RG-LRU state dimension
    use_short_conv: bool = True
    conv_kernel: int = 4

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Position encoding
    rope_base: float = 10000.0
    rope_scaling: Optional[float] = None  # NTK scaling factor

    # Layer norm
    norm_eps: float = 1e-6
    use_rms_norm: bool = True

    # Initialization
    init_std: float = 0.02

    def to_config(self, vocab_size: int = 32000, max_seq_len: int = 2048) -> ChimeraConfig:
        """Convert genome to ChimeraConfig."""
        return ChimeraConfig(
            vocab_size=vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            ffn_hidden=int(self.d_model * self.ffn_multiplier),
            max_seq_len=max_seq_len,
            window_size=self.window_size,
            recurrence_dim=self.recurrence_dim,
            recurrence_expansion=int(self.d_model * 1.5 / self.recurrence_dim),
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            rope_base=self.rope_base,
            norm_eps=self.norm_eps,
        )

    def get_hash(self) -> str:
        """Unique identifier for this genome."""
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:8]

    def estimate_params(self) -> int:
        """Estimate parameter count."""
        d = self.d_model
        n = self.n_layers
        v = 32000  # Assume vocab

        # Embeddings
        params = v * d

        # Per layer (rough estimate)
        ffn = d * int(d * self.ffn_multiplier) * 3  # SwiGLU
        attn = d * d * 4  # Q, K, V, O (simplified)
        rec = d * self.recurrence_dim * 2  # Recurrence

        # Mix based on ratio
        attn_layers = n // (self.recurrence_ratio + 1)
        rec_layers = n - attn_layers

        params += attn_layers * (ffn + attn)
        params += rec_layers * (ffn + rec)

        # Output head
        params += v * d

        return int(params)


# ==============================================================================
# MUTATION OPERATORS
# ==============================================================================

class MutationOperator:
    """Base class for mutation operators."""

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        raise NotImplementedError


class ScaleMutation(MutationOperator):
    """Mutate scale-related genes (dimensions, layers)."""

    def __init__(self, max_param_budget: int = 500_000_000):
        self.max_params = max_param_budget

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        g = copy.deepcopy(genome)

        choice = random.choice(["d_model", "n_layers", "ffn_mult", "heads"])

        if choice == "d_model":
            # Scale d_model by factor
            options = [512, 640, 768, 896, 1024, 1280, 1536, 2048]
            g.d_model = random.choice(options)
            # Keep heads divisible
            while g.d_model % g.n_heads != 0:
                g.n_heads = max(1, g.n_heads - 1)

        elif choice == "n_layers":
            g.n_layers = random.choice([6, 8, 10, 12, 16, 20, 24])

        elif choice == "ffn_mult":
            g.ffn_multiplier = random.choice([2.0, 2.33, 2.67, 3.0, 4.0])

        elif choice == "heads":
            options = [h for h in [4, 6, 8, 12, 16] if g.d_model % h == 0]
            if options:
                g.n_heads = random.choice(options)

        # Enforce param budget
        while g.estimate_params() > self.max_params and g.n_layers > 4:
            g.n_layers -= 1

        return g


class RecurrenceMutation(MutationOperator):
    """Mutate recurrence-related genes."""

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        g = copy.deepcopy(genome)

        choice = random.choice(["ratio", "dim", "conv"])

        if choice == "ratio":
            g.recurrence_ratio = random.choice([1, 2, 3, 4, 5])

        elif choice == "dim":
            g.recurrence_dim = random.choice([256, 384, 512, 768, 1024])

        elif choice == "conv":
            g.use_short_conv = random.choice([True, False])
            if g.use_short_conv:
                g.conv_kernel = random.choice([3, 4, 5, 7])

        return g


class AttentionMutation(MutationOperator):
    """Mutate attention-related genes."""

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        g = copy.deepcopy(genome)

        choice = random.choice(["window", "kv_heads", "rope"])

        if choice == "window":
            g.window_size = random.choice([256, 512, 1024, 2048])

        elif choice == "kv_heads":
            # KV heads must divide n_heads
            options = [h for h in [1, 2, 4, 6, 8] if g.n_heads % h == 0]
            if options:
                g.n_kv_heads = random.choice(options)

        elif choice == "rope":
            g.rope_base = random.choice([10000.0, 50000.0, 100000.0])
            g.rope_scaling = random.choice([None, 2.0, 4.0])

        return g


class RegularizationMutation(MutationOperator):
    """Mutate regularization genes."""

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        g = copy.deepcopy(genome)

        g.dropout = random.choice([0.0, 0.05, 0.1, 0.15])
        g.attention_dropout = random.choice([0.0, 0.05, 0.1])

        return g


class ActivationMutation(MutationOperator):
    """Mutate activation function."""

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        g = copy.deepcopy(genome)
        g.ffn_activation = random.choice(["swiglu", "gelu", "relu"])
        return g


# ==============================================================================
# CROSSOVER
# ==============================================================================

def crossover(parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> ArchitectureGenome:
    """Create child by mixing genes from two parents."""
    child = ArchitectureGenome()

    # For each gene, pick from either parent
    genes = [
        "d_model", "n_layers", "n_heads", "n_kv_heads",
        "ffn_multiplier", "ffn_activation", "recurrence_ratio",
        "window_size", "recurrence_dim", "use_short_conv", "conv_kernel",
        "dropout", "attention_dropout", "rope_base", "rope_scaling",
    ]

    for gene in genes:
        if random.random() < 0.5:
            setattr(child, gene, getattr(parent1, gene))
        else:
            setattr(child, gene, getattr(parent2, gene))

    # Fix compatibility issues
    while child.d_model % child.n_heads != 0:
        child.n_heads = max(1, child.n_heads - 1)
    while child.n_heads % child.n_kv_heads != 0:
        child.n_kv_heads = max(1, child.n_kv_heads - 1)

    return child


# ==============================================================================
# FITNESS EVALUATION
# ==============================================================================

@dataclass
class FitnessResult:
    """Result of evaluating a genome's fitness."""
    genome_hash: str
    loss: float
    perplexity: float
    params: int
    tokens_per_sec: float
    memory_mb: float
    train_time: float

    @property
    def fitness(self) -> float:
        """
        Composite fitness score.

        Balances:
        - Lower loss (primary)
        - Faster training (secondary)
        - Smaller model (tertiary)
        """
        # Normalize components
        loss_score = max(0, 10 - self.loss)  # Lower loss = higher score
        speed_score = min(5, self.tokens_per_sec / 1000)  # tok/s bonus
        size_score = max(0, 3 - self.params / 500_000_000)  # Smaller = better

        return loss_score * 1.0 + speed_score * 0.3 + size_score * 0.2


class FitnessEvaluator:
    """Evaluate genome fitness through short training runs."""

    def __init__(
        self,
        data_path: str,
        tokenizer: ChimeraTokenizer,
        eval_steps: int = 200,
        seq_length: int = 256,
        batch_size: int = 4,
        device: str = "cuda",
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def evaluate(self, genome: ArchitectureGenome) -> FitnessResult:
        """Evaluate a genome with a short training run."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_time = time.time()
        try:
            # Build model
            config = genome.to_config(vocab_size=self.tokenizer.vocab_size)
            model = Chimera(config).to(self.device)

            # Quick memory check
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                memory_mb = 0

            # Dataset
            dataset = PackedDataset(
                self.data_path, self.tokenizer, self.seq_length
            )
            loader = DataLoader(dataset, batch_size=self.batch_size)

            # Optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Short training loop
            model.train()
            total_loss = 0.0
            total_tokens = 0
            data_iter = iter(loader)

            for step in range(self.eval_steps):
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
                total_tokens += input_ids.numel()

            train_time = time.time() - start_time
            avg_loss = total_loss / self.eval_steps
            tokens_per_sec = total_tokens / train_time

            return FitnessResult(
                genome_hash=genome.get_hash(),
                loss=avg_loss,
                perplexity=math.exp(min(avg_loss, 20)),
                params=genome.estimate_params(),
                tokens_per_sec=tokens_per_sec,
                memory_mb=memory_mb,
                train_time=train_time,
            )

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return FitnessResult(
                genome_hash=genome.get_hash(),
                loss=float('inf'),
                perplexity=float('inf'),
                params=genome.estimate_params(),
                tokens_per_sec=0,
                memory_mb=0,
                train_time=time.time() - start_time,
            )

        finally:
            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ==============================================================================
# EVOLUTIONARY SEARCH
# ==============================================================================

@dataclass
class Individual:
    """Individual in the population."""
    genome: ArchitectureGenome
    fitness: Optional[FitnessResult] = None
    generation: int = 0


class EvolutionarySearch:
    """
    Population-based evolutionary architecture search.

    Implements (mu + lambda) evolution strategy with elitism.
    """

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        population_size: int = 10,
        offspring_size: int = 5,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.3,
        elite_size: int = 2,
        max_generations: int = 20,
        output_dir: str = "evolution",
    ):
        self.evaluator = evaluator
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.output_dir = Path(output_dir)

        # Mutation operators
        self.mutators = [
            ScaleMutation(),
            RecurrenceMutation(),
            AttentionMutation(),
            RegularizationMutation(),
            ActivationMutation(),
        ]

        # State
        self.population: List[Individual] = []
        self.generation = 0
        self.history: List[Dict] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize_population(self, seed_genomes: Optional[List[ArchitectureGenome]] = None):
        """Create initial population."""
        if seed_genomes:
            self.population = [Individual(g, generation=0) for g in seed_genomes]

        # Fill remaining with random variations
        base = ArchitectureGenome()
        while len(self.population) < self.population_size:
            genome = copy.deepcopy(base)
            # Apply random mutations
            for _ in range(random.randint(1, 3)):
                mutator = random.choice(self.mutators)
                genome = mutator.mutate(genome)
            self.population.append(Individual(genome, generation=0))

        print(f"Initialized population with {len(self.population)} individuals")

    def evaluate_population(self):
        """Evaluate fitness of all individuals without fitness."""
        for i, ind in enumerate(self.population):
            if ind.fitness is None:
                print(f"  Evaluating individual {i+1}/{len(self.population)} "
                      f"[{ind.genome.get_hash()}]...")
                ind.fitness = self.evaluator.evaluate(ind.genome)
                print(f"    Loss: {ind.fitness.loss:.4f}, "
                      f"PPL: {ind.fitness.perplexity:.2f}, "
                      f"Params: {ind.fitness.params:,}")

    def select_parents(self) -> List[Individual]:
        """Select parents using tournament selection."""
        parents = []
        for _ in range(self.offspring_size):
            # Tournament of 3
            contestants = random.sample(self.population, min(3, len(self.population)))
            winner = max(contestants, key=lambda x: x.fitness.fitness if x.fitness else -float('inf'))
            parents.append(winner)
        return parents

    def create_offspring(self, parents: List[Individual]) -> List[Individual]:
        """Create offspring through mutation and crossover."""
        offspring = []

        for parent in parents:
            child_genome = copy.deepcopy(parent.genome)

            # Crossover with another parent
            if random.random() < self.crossover_rate and len(parents) > 1:
                other = random.choice([p for p in parents if p != parent])
                child_genome = crossover(child_genome, other.genome)

            # Mutation
            if random.random() < self.mutation_rate:
                mutator = random.choice(self.mutators)
                child_genome = mutator.mutate(child_genome)

            offspring.append(Individual(child_genome, generation=self.generation + 1))

        return offspring

    def select_survivors(self):
        """Select survivors for next generation (elitism + tournament)."""
        # Sort by fitness
        self.population.sort(
            key=lambda x: x.fitness.fitness if x.fitness else -float('inf'),
            reverse=True
        )

        # Keep elites
        survivors = self.population[:self.elite_size]

        # Fill rest with tournament selection from remaining
        remaining = self.population[self.elite_size:]
        while len(survivors) < self.population_size and remaining:
            contestants = random.sample(remaining, min(2, len(remaining)))
            winner = max(contestants, key=lambda x: x.fitness.fitness if x.fitness else -float('inf'))
            survivors.append(winner)
            remaining.remove(winner)

        self.population = survivors

    def run(self) -> ArchitectureGenome:
        """Run evolutionary search."""
        print("=" * 60)
        print("EVOLUTIONARY ARCHITECTURE SEARCH")
        print("=" * 60)

        if not self.population:
            self.initialize_population()

        for gen in range(self.max_generations):
            self.generation = gen
            print(f"\n--- Generation {gen + 1}/{self.max_generations} ---")

            # Evaluate
            print("Evaluating population...")
            self.evaluate_population()

            # Record history
            best = max(self.population, key=lambda x: x.fitness.fitness if x.fitness else -float('inf'))
            gen_stats = {
                "generation": gen,
                "best_loss": best.fitness.loss if best.fitness else float('inf'),
                "best_hash": best.genome.get_hash(),
                "best_params": best.genome.estimate_params(),
                "population_size": len(self.population),
            }
            self.history.append(gen_stats)
            print(f"Best: loss={gen_stats['best_loss']:.4f}, "
                  f"params={gen_stats['best_params']:,}")

            # Save checkpoint
            self.save_checkpoint()

            # Create offspring
            parents = self.select_parents()
            offspring = self.create_offspring(parents)

            # Add offspring to population
            self.population.extend(offspring)

            # Evaluate new offspring
            self.evaluate_population()

            # Select survivors
            self.select_survivors()

        # Return best genome
        best = max(self.population, key=lambda x: x.fitness.fitness if x.fitness else -float('inf'))
        print(f"\n{'=' * 60}")
        print(f"EVOLUTION COMPLETE")
        print(f"Best genome: {best.genome.get_hash()}")
        print(f"Loss: {best.fitness.loss:.4f}")
        print(f"Params: {best.genome.estimate_params():,}")
        print(f"{'=' * 60}")

        return best.genome

    def save_checkpoint(self):
        """Save evolution state."""
        data = {
            "generation": self.generation,
            "history": self.history,
            "population": [
                {
                    "genome": asdict(ind.genome),
                    "fitness": asdict(ind.fitness) if ind.fitness else None,
                    "generation": ind.generation,
                }
                for ind in self.population
            ],
        }
        with open(self.output_dir / "evolution_state.json", "w") as f:
            json.dump(data, f, indent=2)

        # Save best genome separately
        best = max(self.population, key=lambda x: x.fitness.fitness if x.fitness else -float('inf'))
        with open(self.output_dir / "best_genome.json", "w") as f:
            json.dump(asdict(best.genome), f, indent=2)

    def load_checkpoint(self, path: str):
        """Load evolution state."""
        with open(path) as f:
            data = json.load(f)

        self.generation = data["generation"]
        self.history = data["history"]
        self.population = []

        for item in data["population"]:
            genome = ArchitectureGenome(**item["genome"])
            fitness = FitnessResult(**item["fitness"]) if item["fitness"] else None
            self.population.append(Individual(genome, fitness, item["generation"]))

        print(f"Loaded checkpoint from generation {self.generation}")


# ==============================================================================
# MAIN
# ==============================================================================

def run_evolution(
    data_path: str = "data/tinystories.txt",
    output_dir: str = "evolution",
    population_size: int = 8,
    max_generations: int = 10,
    eval_steps: int = 150,
):
    """Run evolutionary architecture search."""
    print("Initializing evolutionary search...")

    # Tokenizer
    tokenizer = ChimeraTokenizer()

    # Evaluator
    evaluator = FitnessEvaluator(
        data_path=data_path,
        tokenizer=tokenizer,
        eval_steps=eval_steps,
        seq_length=256,
        batch_size=4,
    )

    # Seed genomes (start from known good configs)
    seeds = [
        ArchitectureGenome(),  # Default (small-ish)
        ArchitectureGenome(d_model=512, n_layers=8, n_heads=8),  # Smaller
        ArchitectureGenome(d_model=1024, n_layers=12, n_heads=16, n_kv_heads=4),  # Larger
        ArchitectureGenome(recurrence_ratio=5, window_size=256),  # More recurrent
        ArchitectureGenome(recurrence_ratio=1, window_size=1024),  # More attention
    ]

    # Evolution
    search = EvolutionarySearch(
        evaluator=evaluator,
        population_size=population_size,
        offspring_size=population_size // 2,
        max_generations=max_generations,
        output_dir=output_dir,
    )

    search.initialize_population(seeds)
    best_genome = search.run()

    # Save best config for training
    config = best_genome.to_config(vocab_size=tokenizer.vocab_size)
    print(f"\nBest architecture saved to {output_dir}/best_genome.json")
    print(f"To train: python train_packed.py --config-file {output_dir}/best_genome.json")

    return best_genome


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evolutionary Architecture Search")
    parser.add_argument("--data-path", type=str, default="data/tinystories.txt")
    parser.add_argument("--output-dir", type=str, default="evolution")
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=150)

    args = parser.parse_args()

    run_evolution(
        data_path=args.data_path,
        output_dir=args.output_dir,
        population_size=args.population_size,
        max_generations=args.max_generations,
        eval_steps=args.eval_steps,
    )
