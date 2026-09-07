import argparse
import torch
import torch.nn as nn

from pso import Optimizer
from cli import add_pso_args, build_optimizer_kwargs


def get_data(seed: int = 42):
    from sklearn.decomposition import PCA
    from torchvision.datasets import MNIST

    train_dataset = MNIST(root="./data", train=True, download=True)
    test_dataset = MNIST(root="./data", train=False, download=True)

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
    parser = argparse.ArgumentParser(description="PSO MNIST Benchmark Script")
    add_pso_args(
        parser,
        defaults={
            "method": "inertia",
            "initialization": "model_noise",
            "evaluation": "fixed_subset",
            "convergence": "none",
            "refinement": "adam",
            "n_particles": 30,
            "c0": None,
            "c1": None,
            "w_min": None,
            "w_max": None,
            "negative_swarm": 0.0,
            "mutation_swarm": 0.02,
            "particle_min": -3.0,
            "particle_max": 3.0,
            "velocity_limit_ratio": 0.025,
            "boundary_strategy": "reflect",
            "initial_position_noise": 0.05,
            "seed": 42,
            "epochs": 80,
            "batch_size": 1000,
            "fitness_size": 2000,
            "renewal": "loss",
            "output_dir": "output/mnist",
            "checkpoint_interval": 25,
            "refinement_epochs": 100,
            "refinement_lr": 0.01,
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
        inertia_profile={"c0": 1.49618, "c1": 1.49618, "w_min": 0.7298, "w_max": 0.7298},
    )
    pso_mnist = Optimizer(**kwargs)

    print(f"Optimizer device: {pso_mnist.device}")

    best_score = pso_mnist.fit(
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
