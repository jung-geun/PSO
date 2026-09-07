import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pso import Optimizer
from cli import add_pso_args, build_optimizer_kwargs


def get_data(seed: int = 42):
    with open("data/seeds/seeds_dataset.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
        df = pd.DataFrame([d.split() for d in data])
        df.columns = [
            "area",
            "perimeter",
            "compactness",
            "length_of_kernel",
            "width_of_kernel",
            "asymmetry_coefficient",
            "length_of_kernel_groove",
            "target",
        ]

        df = df.astype(float)
        df["target"] = df["target"].astype(int) - 1

        x = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.int64),
    )


def make_model(seed: int = 42):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(7, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )


def main():
    parser = argparse.ArgumentParser(description="PSO Seeds Benchmark Script")
    add_pso_args(
        parser,
        defaults={
            "method": "original",
            "initialization": "model_noise",
            "evaluation": "full",
            "convergence": "particle_reset",
            "refinement": "adam",
            "n_particles": 24,
            "c0": None,
            "c1": None,
            "w_min": None,
            "w_max": None,
            "negative_swarm": 0.0,
            "mutation_swarm": 0.3,
            "particle_min": -3.0,
            "particle_max": 3.0,
            "velocity_limit_ratio": 0.1,
            "boundary_strategy": "reflect",
            "seed": 42,
            "epochs": 80,
            "renewal": "acc",
            "output_dir": "output/seeds",
            "checkpoint_interval": 25,
            "refinement_epochs": 10,
            "refinement_lr": 0.001,
        },
    )
    args = parser.parse_args()

    model = make_model(seed=args.seed)
    x_train, y_train, x_test, y_test = get_data(seed=args.seed)

    fitness_size = args.fitness_size if args.evaluation == "fixed_subset" else None
    refinement_epochs = args.refinement_epochs if args.refinement == "adam" else 0

    kwargs = build_optimizer_kwargs(
        args,
        model=model,
        loss=nn.CrossEntropyLoss(),
        task="multiclass",
        inertia_profile={"c0": 0.5, "c1": 1.0, "w_min": 0.7, "w_max": 1.2},
    )
    pso_seeds = Optimizer(**kwargs)

    print(f"Optimizer device: {pso_seeds.device}")

    best_score = pso_seeds.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fitness_size=fitness_size,
        renewal=args.renewal,
        validation_data=(x_test, y_test),
        output_dir=args.output_dir,
        checkpoint_interval=25,
        save_info=True,
        refinement_epochs=refinement_epochs,
        refinement_lr=args.refinement_lr,
    )

    print(f"Done! Best score: {best_score}")


if __name__ == "__main__":
    main()
