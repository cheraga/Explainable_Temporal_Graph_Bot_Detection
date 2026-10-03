
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from evaluation import evaluate


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_split(labels, seed):
    """
    Create stratified train/validation/test masks.

    Split:
        70% train
        20% validation
        10% test
    """

    indices = np.arange(
        len(labels)
    )

    labels_numpy = labels.cpu().numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=labels_numpy
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 / 3,
        random_state=seed,
        stratify=labels_numpy[temp_idx]
    )

    train_mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    val_mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    test_mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return (
        train_mask,
        val_mask,
        test_mask
    )


def train_model(
    model,
    x,
    y,
    edge_index,
    edge_type,
    train_mask,
    val_mask,
    test_mask,
    epochs,
    learning_rate,
    weight_decay
):
    """
    Train one model and evaluate it on the test set.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    if edge_type is not None:
        edge_type = edge_type.to(device)

    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    best_state = None

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        if edge_type is None:

            output = model(
                x,
                edge_index
            )

        else:

            output = model(
                x,
                edge_index,
                edge_type
            )

        loss = criterion(
            output[train_mask],
            y[train_mask]
        )

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            if edge_type is None:

                val_output = model(
                    x,
                    edge_index
                )

            else:

                val_output = model(
                    x,
                    edge_index,
                    edge_type
                )

            val_prediction = (
                val_output.argmax(
                    dim=1
                )
            )

            val_accuracy = (
                val_prediction[val_mask]
                == y[val_mask]
            ).float().mean().item()

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    model = model.to(device)

    model.eval()

    with torch.no_grad():

        if edge_type is None:

            output = model(
                x,
                edge_index
            )

        else:

            output = model(
                x,
                edge_index,
                edge_type
            )

        probability = torch.softmax(
            output,
            dim=1
        )

        prediction = output.argmax(
            dim=1
        )

    y_true = (
        y[test_mask]
        .cpu()
        .numpy()
    )

    y_pred = (
        prediction[test_mask]
        .cpu()
        .numpy()
    )

    y_probability = (
        probability[test_mask, 1]
        .cpu()
        .numpy()
    )

    results = evaluate(
        y_true,
        y_pred,
        y_probability
    )

    return results
