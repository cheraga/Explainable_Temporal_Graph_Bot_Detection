
import os
import sys

import numpy as np
import pandas as pd
import torch


# Add src/ to the Python path
SRC_PATH = os.path.join(
    os.path.dirname(__file__),
    "src"
)

sys.path.append(SRC_PATH)


from data_loader import load_mgtab
from graph_builder import (
    select_relations,
    graph_statistics
)
from features import (
    clean_features,
    fit_standardizer,
    apply_standardizer
)
from models import (
    GCNBaseline,
    RGCNBaseline,
    MultiRelationalAttentionModel
)
from train import (
    set_seed,
    create_split,
    train_model
)

import config


DATASET_PATH = "Dataset/MGTAB"


def run_experiment(
    model_name,
    seed,
    features,
    labels,
    edge_index,
    edge_type
):
    """
    Run one experiment for one model and one seed.
    """

    print("\n" + "=" * 60)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Seed: {seed}"
    )

    print("=" * 60)

    set_seed(seed)

    train_mask, val_mask, test_mask = create_split(
        labels,
        seed
    )

    # Fit feature normalization ONLY on training nodes
    mean, std = fit_standardizer(
        features,
        train_mask
    )

    normalized_features = apply_standardizer(
        features,
        mean,
        std
    )

    input_dim = normalized_features.shape[1]

    if model_name == "GCN":

        model = GCNBaseline(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )

        model_edge_type = None

    elif model_name == "RGCN":

        model = RGCNBaseline(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_relations=len(config.RELATIONS),
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )

        model_edge_type = edge_type

    elif model_name == "MultiRelationalAttention":

        model = MultiRelationalAttentionModel(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_relations=len(config.RELATIONS),
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )

        model_edge_type = edge_type

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    results = train_model(
        model=model,
        x=normalized_features,
        y=labels,
        edge_index=edge_index,
        edge_type=model_edge_type,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    print("\nResults:")

    for key, value in results.items():

        if key != "confusion_matrix":

            print(
                f"{key}: {value}"
            )

    print("\nConfusion Matrix:")

    print(
        results["confusion_matrix"]
    )

    return results


def main():

    print("=" * 60)

    print(
        "Explainable Graph-Based Bot Detection"
    )

    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    (
        features,
        labels,
        edge_index,
        edge_type,
        edge_weight
    ) = load_mgtab(
        DATASET_PATH
    )

    # --------------------------------------------------
    # 2. Basic validation
    # --------------------------------------------------

    if len(labels) != features.shape[0]:

        raise ValueError(
            "Number of labels does not match "
            "number of nodes."
        )

    if edge_index.shape[0] != 2:

        raise ValueError(
            "edge_index must have shape [2, E]."
        )

    if edge_type.shape[0] != edge_index.shape[1]:

        raise ValueError(
            "edge_type length does not match "
            "number of edges."
        )

    # --------------------------------------------------
    # 3. Clean features
    # --------------------------------------------------

    features = clean_features(
        features
    )

    # --------------------------------------------------
    # 4. Select relations
    # --------------------------------------------------

    (
        edge_index,
        edge_type,
        edge_weight
    ) = select_relations(
        edge_index,
        edge_type,
        edge_weight,
        config.RELATION_SELECTION
    )

    # --------------------------------------------------
    # 5. Graph statistics
    # --------------------------------------------------

    graph_statistics(
        edge_index,
        edge_type
    )

    # --------------------------------------------------
    # 6. Experiments
    # --------------------------------------------------

    model_names = [
        "GCN",
        "RGCN",
        "MultiRelationalAttention"
    ]

    all_results = []

    for model_name in model_names:

        for seed in config.SEEDS:

            results = run_experiment(
                model_name=model_name,
                seed=seed,
                features=features,
                labels=labels,
                edge_index=edge_index,
                edge_type=edge_type
            )

            row = {
                "model": model_name,
                "seed": seed,
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
                "roc_auc": results["roc_auc"]
            }

            all_results.append(
                row
            )

    # --------------------------------------------------
    # 7. Save results
    # --------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        "results/"
        "baseline_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 60)

    print(
        f"Results saved to: {results_path}"
    )

    print("=" * 60)

    # --------------------------------------------------
    # 8. Summary
    # --------------------------------------------------

    summary = (
        results_df
        .groupby("model")
        [
            [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"
            ]
        ]
        .agg(["mean", "std"])
    )

    print("\nSummary:")

    print(summary)


if __name__ == "__main__":
    main()
