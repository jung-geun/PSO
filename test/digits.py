import argparse
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pso import Optimizer
from cli import add_pso_args, build_optimizer_kwargs


def make_model(seed: int = 42):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 12),
        nn.ReLU(),
        nn.Linear(12, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
    )


def get_data(seed: int = 42):
    digits = load_digits()
    x = digits.data.astype("float32")
    y = digits.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, shuffle=True
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
        torch.tensor(y_test, dtype=torch.int64),
    )


def main():
    parser = argparse.ArgumentParser(description="PSO Digits Benchmark Script")
    add_pso_args(
        parser,
        defaults={
            "method": "original",
            "initialization": "model_noise",
            "evaluation": "fixed_subset",
            "convergence": "particle_reset",
            "refinement": "adam",
            "n_particles": 30,
            "c0": None,
            "c1": None,
            "w_min": None,
            "w_max": None,
            "negative_swarm": 0.0,
            "mutation_swarm": 0.1,
            "particle_min": -3.0,
            "particle_max": 3.0,
            "velocity_limit_ratio": 0.1,
            "boundary_strategy": "reflect",
            "seed": 42,
            "epochs": 80,
            "batch_size": 200,
            "fitness_size": 1000,
            "renewal": "loss",
            "output_dir": "output/digits",
            "refinement_epochs": 10,
            "refinement_lr": 0.001,
        },
    )
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = get_data(seed=args.seed)
    model = make_model(seed=args.seed)

    fitness_size = args.fitness_size if args.evaluation == "fixed_subset" else None
    refinement_epochs = args.refinement_epochs if args.refinement == "adam" else 0

    kwargs = build_optimizer_kwargs(
        args,
        model=model,
        loss=nn.CrossEntropyLoss(),
        task="multiclass",
        inertia_profile={"c0": 0.5, "c1": 0.3, "w_min": 0.2, "w_max": 0.9},
    )
    digits_pso = Optimizer(**kwargs)

    print(f"Optimizer device: {digits_pso.device}")

    best_score = digits_pso.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fitness_size=fitness_size,
        renewal=args.renewal,
        validation_data=(x_test, y_test),
        output_dir=args.output_dir,
        save_info=True,
        refinement_epochs=refinement_epochs,
        refinement_lr=args.refinement_lr,
    )

    print(f"Done! Best score: {best_score}")


if __name__ == "__main__":
    main()
