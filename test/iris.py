import argparse
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pso import Optimizer
from cli import add_pso_args, build_optimizer_kwargs


def make_model(seed: int = 42):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )


def load_data(seed: int = 42):
    iris = load_iris()
    x = iris.data.astype("float32")
    y = iris.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, stratify=y, random_state=seed
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
    parser = argparse.ArgumentParser(description="PSO Iris Benchmark Script")
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
            "mutation_swarm": 0.1,
            "particle_min": -3.0,
            "particle_max": 3.0,
            "velocity_limit_ratio": 0.1,
            "boundary_strategy": "reflect",
            "seed": 42,
            "epochs": 70,
            "renewal": "loss",
            "output_dir": "output/iris",
            "checkpoint_interval": 25,
            "refinement_epochs": 10,
            "refinement_lr": 0.001,
        },
    )
    args = parser.parse_args()

    model = make_model(seed=args.seed)
    x_train, x_test, y_train, y_test = load_data(seed=args.seed)

    fitness_size = args.fitness_size if args.evaluation == "fixed_subset" else None
    refinement_epochs = args.refinement_epochs if args.refinement == "adam" else 0

    kwargs = build_optimizer_kwargs(
        args,
        model=model,
        loss=nn.CrossEntropyLoss(),
        task="multiclass",
        inertia_profile={"c0": 0.5, "c1": 0.3, "w_min": 0.1, "w_max": 0.9},
    )
    pso_iris = Optimizer(**kwargs)

    print(f"Optimizer device: {pso_iris.device}")

    best_score = pso_iris.fit(
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
