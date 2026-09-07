import argparse
import torch
import torch.nn as nn

from pso import Optimizer
from cli import add_pso_args, build_optimizer_kwargs


def get_data():
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    return x, y


def make_model(seed: int = 101):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    )


def main():
    parser = argparse.ArgumentParser(description="PSO XOR Benchmark Script")
    add_pso_args(
        parser,
        defaults={
            "method": "original",
            "initialization": "model_noise",
            "evaluation": "fixed_subset",
            "convergence": "none",
            "refinement": "adam",
            "n_particles": 40,
            "c0": None,
            "c1": None,
            "w_min": None,
            "w_max": None,
            "negative_swarm": 0.1,
            "mutation_swarm": 0.03,
            "particle_min": -5.0,
            "particle_max": 5.0,
            "velocity_limit_ratio": 0.1,
            "boundary_strategy": "reflect",
            "initial_position_noise": 1.0,
            "seed": 101,
            "epochs": 120,
            "fitness_size": 4,
            "renewal": "loss",
            "output_dir": "output/xor",
            "refinement_epochs": 100,
            "refinement_lr": 0.03,
        },
    )
    args = parser.parse_args()
    x, y = get_data()
    model = make_model(seed=args.seed)

    fitness_size = args.fitness_size if args.evaluation == "fixed_subset" else None
    refinement_epochs = args.refinement_epochs if args.refinement == "adam" else 0

    kwargs = build_optimizer_kwargs(
        args,
        model=model,
        loss=nn.BCEWithLogitsLoss(),
        task="binary",
        inertia_profile={"c0": 0.7, "c1": 0.9, "w_min": 0.3, "w_max": 0.8},
    )
    pso_xor = Optimizer(**kwargs)
    print(f"Optimizer device: {pso_xor.device}")

    best_score = pso_xor.fit(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fitness_size=fitness_size,
        renewal=args.renewal,
        output_dir=args.output_dir,
        save_info=True,
        refinement_epochs=refinement_epochs,
        refinement_lr=args.refinement_lr,
    )

    print(f"Done! Best score: {best_score}")


if __name__ == "__main__":
    main()
