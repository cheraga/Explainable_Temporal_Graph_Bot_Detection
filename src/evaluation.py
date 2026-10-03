import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate(
    y_true,
    y_pred,
    y_probability
):
    """
    Calculate classification metrics.
    """

    results = {}

    results["accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    results["precision"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    results["recall"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    results["f1"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    try:
        results["roc_auc"] = roc_auc_score(
            y_true,
            y_probability
        )
    except ValueError:
        results["roc_auc"] = np.nan

    results["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred
    )

    return results
