import torch


def clean_features(features):
    """
    Replace NaN and infinite values with zero.
    """

    features = features.float()

    features = torch.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


def fit_standardizer(features, train_mask):
    """
    Calculate mean and standard deviation
    using ONLY the training nodes.

    This avoids feature normalization leakage
    from validation/test nodes.
    """

    train_features = features[train_mask]

    mean = train_features.mean(
        dim=0,
        keepdim=True
    )

    std = train_features.std(
        dim=0,
        keepdim=True
    )

    std[std == 0] = 1.0

    return mean, std


def apply_standardizer(features, mean, std):
    """
    Apply a previously fitted standardizer.
    """

    return (features - mean) / std
