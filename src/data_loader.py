import os
import torch


REQUIRED_FILES = [
    "edge_index.pt",
    "edge_type.pt",
    "edge_weight.pt",
    "features.pt",
    "labels_bot.pt",
]


def check_dataset(dataset_path):
    """
    Check whether all required MGTAB files exist.
    """

    print("Checking MGTAB dataset...")

    missing_files = []

    for filename in REQUIRED_FILES:
        file_path = os.path.join(dataset_path, filename)

        if not os.path.exists(file_path):
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(
            "The following MGTAB files are missing:\n"
            + "\n".join(missing_files)
        )

    print("All required MGTAB files found.")


def load_mgtab(dataset_path):
    """
    Load MGTAB tensors.

    Returns:
        features
        labels
        edge_index
        edge_type
        edge_weight
    """

    check_dataset(dataset_path)

    edge_index = torch.load(
        os.path.join(dataset_path, "edge_index.pt")
    )

    edge_type = torch.load(
        os.path.join(dataset_path, "edge_type.pt")
    )

    edge_weight = torch.load(
        os.path.join(dataset_path, "edge_weight.pt")
    )

    features = torch.load(
        os.path.join(dataset_path, "features.pt")
    )

    labels = torch.load(
        os.path.join(dataset_path, "labels_bot.pt")
    )

    # Make sure tensors have the expected types/shapes
    features = features.float()
    labels = labels.view(-1).long()
    edge_index = edge_index.long()
    edge_type = edge_type.view(-1).long()
    edge_weight = edge_weight.float()

    print("\nDataset loaded successfully.")

    print("Number of nodes:", features.shape[0])
    print("Number of features:", features.shape[1])
    print("Number of edges:", edge_index.shape[1])

    print("\nLabel distribution:")

    unique_labels, counts = torch.unique(
        labels,
        return_counts=True
    )

    for label, count in zip(
        unique_labels.tolist(),
        counts.tolist()
    ):
        print(f"Label {label}: {count}")

    return (
        features,
        labels,
        edge_index,
        edge_type,
        edge_weight,
    )
