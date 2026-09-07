import argparse
import torch
import torch.nn as nn

from pso import Optimizer
from cli import add_pso_args, build_optimizer_kwargs


def get_data(seed: int = 42):
    from sklearn.decomposition import PCA
    from torchvision.datasets import FashionMNIST

    train_dataset = FashionMNIST(root="./data", train=True, download=True)
    test_dataset = FashionMNIST(root="./data", train=False, download=True)

    x_train_raw = (train_dataset.data[:3000].float() / 255.0).reshape(3000, -1).numpy()
    y_train = train_dataset.targets[:3000].long()

    x_test_raw = (test_dataset.data[:1000].float() / 255.0).reshape(1000, -1).numpy()
    y_test = test_dataset.targets[:1000].long()

    pca = PCA(n_components=32, whiten=True, random_state=seed)
    x_train_pca = pca.fit_transform(x_train_raw)
    x_test_pca = pca.transform(x_test_raw)

    x_train = torch.tensor(x_train_pca, dtype=torch.float32)
    x_test = torch.tensor(x_test_pca, dtype=torch.float32)

    print(f"x_train : {x_train.shape} | y_train : {y_train.shape}")
    print(f"x_test : {x_test.shape} | y_test : {y_test.shape}")

    return x_train, y_train, x_test, y_test


def make_model(seed: int = 42):
    torch.manual_seed(seed)
    return nn.Linear(32, 10)


def main():
    parser = argparse.ArgumentParser(description="PSO Fashion-MNIST Benchmark Script")
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
            "mutation_swarm": 0.05,
            "particle_min": -3.0,
            "particle_max": 3.0,
            "velocity_limit_ratio": 0.1,
            "boundary_strategy": "reflect",
            "seed": 42,
            "epochs": 80,
            "batch_size": 1000,
            "fitness_size": 2000,
            "renewal": "loss",
            "output_dir": "output/fashion_mnist",
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
        inertia_profile={"c0": 0.7, "c1": 0.5, "w_min": 0.1, "w_max": 0.8},
    )
    pso_fashion = Optimizer(**kwargs)

    print(f"Optimizer device: {pso_fashion.device}")

    best_score = pso_fashion.fit(
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
