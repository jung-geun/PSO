#!/usr/bin/env python3
"""
PSO Stage Plugin Method Comparison Tool.

Compares PSO movement methods (original, inertia, constriction, fips, clpso, bare_bones,
adaptive_moment, local_best, quantum)
across benchmark datasets (xor, iris, mnist) with reproducible model initialization, dataset splits,
and clean console output & JSON result reporting.
"""

import argparse
import json
import sys
import time
from typing import Any

import torch
import torch.nn as nn
from cli import add_pso_args, parse_method_options
from pso import Optimizer
from pso.plugins import available_plugins


def print_available_methods():
    """Prints available movement method plugins and metadata provenance."""
    movement_plugins = available_plugins("movement")
    print("Available PSO Movement Method Plugins:")
    print("=" * 80)
    for name, meta in movement_plugins.items():
        print(f"  Stage Key    : {name}")
        print(f"  Title        : {meta.title}")
        print(f"  Source (DOI) : {meta.source or 'N/A'}")
        print(f"  Fidelity     : {meta.fidelity}")
        print(f"  Needs Grad   : {meta.gradient_required}")
        print("-" * 80)


def get_xor_workload(seed: int):
    """Builds identical XOR dataset and PyTorch model state for a given seed."""
    torch.manual_seed(seed)
    x_train = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
    )
    y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    return x_train, y_train, None, model, loss_fn, "binary"


def get_iris_workload(seed: int):
    """Builds identical Iris dataset and PyTorch model state for a given seed."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    iris = load_iris()
    X = iris.data.astype("float32")
    y = iris.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_tr = torch.tensor(x_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.int64)
    x_te = torch.tensor(x_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.int64)

    model = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )
    loss_fn = nn.CrossEntropyLoss()
    return x_tr, y_tr, (x_te, y_te), model, loss_fn, "multiclass"


def get_mnist_workload(seed: int):
    """Builds identical PCA32 MNIST dataset and PyTorch model state for a given seed."""
    from sklearn.decomposition import PCA
    from torchvision.datasets import MNIST

    torch.manual_seed(seed)
    train_ds = MNIST(root="./data", train=True, download=True)
    test_ds = MNIST(root="./data", train=False, download=True)

    x_tr_raw = (train_ds.data[:3000].float() / 255.0).reshape(3000, -1).numpy()
    y_tr = train_ds.targets[:3000].long()
    x_te_raw = (test_ds.data[:1000].float() / 255.0).reshape(1000, -1).numpy()
    y_te = test_ds.targets[:1000].long()

    pca = PCA(n_components=32, whiten=True, random_state=seed)
    x_tr_pca = pca.fit_transform(x_tr_raw)
    x_te_pca = pca.transform(x_te_raw)

    x_tr = torch.tensor(x_tr_pca, dtype=torch.float32)
    x_te = torch.tensor(x_te_pca, dtype=torch.float32)

    model = nn.Linear(32, 10)
    loss_fn = nn.CrossEntropyLoss()
    return x_tr, y_tr, (x_te, y_te), model, loss_fn, "multiclass"


DATASET_LOADERS = {
    "xor": get_xor_workload,
    "iris": get_iris_workload,
    "mnist": get_mnist_workload,
}


def main():
    parser = argparse.ArgumentParser(
        description="PSO Stage Plugin Method Comparison Surface"
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List available movement methods and exit",
    )
    parser.add_argument(
        "--dataset",
        choices=["xor", "iris", "mnist"],
        default="xor",
        help="Target dataset workload (default: %(default)s)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Movement method keys to evaluate or 'all' (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seed list (default: 42)",
    )
    parser.add_argument(
        "--json-path",
        "--output-json",
        "--json",
        dest="json_path",
        type=str,
        default=None,
        help="Optional JSON file path to save detailed evaluation metrics",
    )

    # Add standard stage selector and hyperparameter options
    add_pso_args(parser, defaults={"n_particles": 20, "epochs": 30})
    args = parser.parse_args()

    if args.list_methods:
        print_available_methods()
        sys.exit(0)

    # Determine movement methods to test
    available_m_plugins = available_plugins("movement")
    if "all" in args.methods or "ALL" in args.methods:
        methods_to_test = list(available_m_plugins.keys())
    else:
        methods_to_test = []
        for m in args.methods:
            if m not in available_m_plugins:
                raise ValueError(
                    f"Unknown movement method '{m}'. Available: {list(available_m_plugins.keys())}"
                )
            methods_to_test.append(m)

    loader = DATASET_LOADERS[args.dataset]
    eval_stage = args.evaluation
    fitness_size = args.fitness_size if eval_stage == "fixed_subset" else None

    if eval_stage == "fixed_subset" and fitness_size is None:
        if args.dataset == "xor":
            fitness_size = 4
        elif args.dataset == "iris":
            fitness_size = 100
        elif args.dataset == "mnist":
            fitness_size = 2000
    refine_stage = args.refinement
    refinement_epochs = args.refinement_epochs if refine_stage == "adam" else 0

    results: list[dict[str, Any]] = []
    parsed_method_opts = parse_method_options(args.method_options)
    has_val = args.dataset != "xor"

    print("=" * 80)
    print(f"PSO Movement Method Comparison on '{args.dataset}' Dataset")
    print(
        f"Stages: initialization='{args.initialization}', evaluation='{eval_stage}', "
        f"convergence='{args.convergence}', refinement='{refine_stage}'"
    )
    print(
        f"Parameters: particles={args.n_particles}, epochs={args.epochs}, seeds={args.seeds}"
    )
    print("=" * 80)
    if has_val:
        print(
            f"{'Method':<18} {'Seed':<6} {'Val Loss':<12} {'Val Accuracy':<12} {'Val MSE':<12} {'Time (s)':<10}"
        )
    else:
        print(
            f"{'Method':<18} {'Seed':<6} {'Loss':<12} {'Accuracy':<12} {'MSE':<12} {'Time (s)':<10}"
        )
    print("-" * 80)

    for method_key in methods_to_test:
        for seed in args.seeds:
            x_tr, y_tr, val_data, model, loss_fn, task = loader(seed)

            eff_fitness_size = (
                min(fitness_size, x_tr.shape[0])
                if fitness_size is not None
                else None
            )

            c0 = args.c0
            c1 = args.c1
            w_min = args.w_min
            w_max = args.w_max

            neg_swarm = (
                args.negative_swarm
                if method_key not in ("fips", "clpso", "bare_bones")
                else 0.0
            )
            mut_swarm = args.mutation_swarm if method_key != "bare_bones" else 0.0
            vel_ratio = (
                args.velocity_limit_ratio if method_key != "bare_bones" else None
            )

            kwargs: dict[str, Any] = {
                "model": model,
                "loss": loss_fn,
                "task": task,
                "method": method_key,
                "initialization": args.initialization,
                "evaluation": eval_stage,
                "convergence": args.convergence,
                "refinement": refine_stage,
                "method_options": parsed_method_opts,
                "n_particles": args.n_particles,
                "c0": c0,
                "c1": c1,
                "w_min": w_min,
                "w_max": w_max,
                "negative_swarm": neg_swarm,
                "mutation_swarm": mut_swarm,
                "particle_min": args.particle_min,
                "particle_max": args.particle_max,
                "velocity_limit_ratio": vel_ratio,
                "boundary_strategy": args.boundary_strategy,
                "initial_position_noise": args.initial_position_noise,
                "seed": seed,
                "device": args.device,
                "fitness_size": eff_fitness_size,
                "convergence_patience": args.convergence_patience,
                "convergence_min_delta": args.convergence_min_delta,
                "convergence_monitor": args.convergence_monitor,
                "refinement_epochs": refinement_epochs,
                "refinement_lr": args.refinement_lr,
            }

            if method_key == "adaptive_moment":
                if args.moment_blend is not None:
                    kwargs["moment_blend"] = args.moment_blend
                elif "moment_blend" in parsed_method_opts:
                    kwargs["moment_blend"] = float(parsed_method_opts["moment_blend"])
                else:
                    kwargs["moment_blend"] = 0.25

                if args.moment_beta1 is not None:
                    kwargs["moment_beta1"] = args.moment_beta1
                if args.moment_beta2 is not None:
                    kwargs["moment_beta2"] = args.moment_beta2
                if args.moment_step_size is not None:
                    kwargs["moment_step_size"] = args.moment_step_size
                if args.moment_epsilon is not None:
                    kwargs["moment_epsilon"] = args.moment_epsilon

            opt = Optimizer(**kwargs)

            start_t = time.perf_counter()
            best_score = opt.fit(
                x_tr,
                y_tr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                fitness_size=eff_fitness_size,
                renewal=args.renewal,
                validation_data=val_data,
                output_dir=None,
                refinement_epochs=refinement_epochs,
                refinement_lr=args.refinement_lr,
            )
            elapsed_t = time.perf_counter() - start_t

            tr_loss, tr_acc, tr_mse = best_score
            if val_data is not None:
                val_x, val_y = val_data
                val_score = opt.evaluate(val_x, val_y)
                eval_loss, eval_acc, eval_mse = val_score
                score_src = "validation"
            else:
                eval_loss, eval_acc, eval_mse = tr_loss, tr_acc, tr_mse
                score_src = "training"

            results.append(
                {
                    "method": method_key,
                    "seed": seed,
                    "score_source": score_src,
                    "train_loss": tr_loss,
                    "train_accuracy": tr_acc,
                    "train_mse": tr_mse,
                    "eval_loss": eval_loss,
                    "eval_accuracy": eval_acc,
                    "eval_mse": eval_mse,
                    "loss": eval_loss,
                    "accuracy": eval_acc,
                    "mse": eval_mse,
                    "elapsed_time": elapsed_t,
                }
            )

            print(
                f"{method_key:<18} {seed:<6} {eval_loss:<12.6f} {eval_acc:<12.6f} {eval_mse:<12.6f} {elapsed_t:<10.4f}"
            )

    print("-" * 80)
    print("\nAggregate Summary (Mean across seeds):")
    print("=" * 80)
    if has_val:
        print(
            f"{'Method':<18} {'Mean Val Loss':<14} {'Mean Val Acc':<14} {'Mean Val MSE':<14} {'Mean Time (s)':<12}"
        )
    else:
        print(
            f"{'Method':<18} {'Mean Loss':<12} {'Mean Acc':<12} {'Mean MSE':<12} {'Mean Time (s)':<12}"
        )
    print("-" * 80)

    summary_list: list[dict[str, Any]] = []
    for method_key in methods_to_test:
        method_runs = [r for r in results if r["method"] == method_key]
        if not method_runs:
            continue
        n_runs = len(method_runs)
        mean_tr_loss = sum(r["train_loss"] for r in method_runs) / n_runs
        mean_tr_acc = sum(r["train_accuracy"] for r in method_runs) / n_runs
        mean_tr_mse = sum(r["train_mse"] for r in method_runs) / n_runs
        mean_eval_loss = sum(r["eval_loss"] for r in method_runs) / n_runs
        mean_eval_acc = sum(r["eval_accuracy"] for r in method_runs) / n_runs
        mean_eval_mse = sum(r["eval_mse"] for r in method_runs) / n_runs
        mean_time = sum(r["elapsed_time"] for r in method_runs) / n_runs
        score_src = method_runs[0]["score_source"]

        summary_entry = {
            "method": method_key,
            "title": available_m_plugins[method_key].title,
            "source": available_m_plugins[method_key].source,
            "score_source": score_src,
            "mean_train_loss": mean_tr_loss,
            "mean_train_accuracy": mean_tr_acc,
            "mean_train_mse": mean_tr_mse,
            "mean_eval_loss": mean_eval_loss,
            "mean_eval_accuracy": mean_eval_acc,
            "mean_eval_mse": mean_eval_mse,
            "mean_loss": mean_eval_loss,
            "mean_accuracy": mean_eval_acc,
            "mean_mse": mean_eval_mse,
            "mean_elapsed_time": mean_time,
            "runs": n_runs,
        }
        summary_list.append(summary_entry)

        if has_val:
            print(
                f"{method_key:<18} {mean_eval_loss:<14.6f} {mean_eval_acc:<14.6f} {mean_eval_mse:<14.6f} {mean_time:<12.4f}"
            )
        else:
            print(
                f"{method_key:<18} {mean_eval_loss:<12.6f} {mean_eval_acc:<12.6f} {mean_eval_mse:<12.6f} {mean_time:<12.4f}"
            )
    print("=" * 80)

    if args.json_path:
        payload = {
            "dataset": args.dataset,
            "score_source": "validation" if has_val else "training",
            "selectors": {
                "initialization": args.initialization,
                "evaluation": eval_stage,
                "convergence": args.convergence,
                "refinement": refine_stage,
            },
            "parameters": {
                "n_particles": args.n_particles,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "fitness_size": fitness_size,
                "refinement_epochs": refinement_epochs,
                "refinement_lr": args.refinement_lr,
                "seeds": args.seeds,
            },
            "results": results,
            "summary": summary_list,
        }
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved comparison results JSON to: {args.json_path}")


if __name__ == "__main__":
    main()
