import torch


def select_relations(
    edge_index,
    edge_type,
    edge_weight,
    relation_selection
):
    """
    Select only the requested relation types.
    """

    mask = torch.zeros(
        edge_type.shape,
        dtype=torch.bool
    )

    for relation in relation_selection:
        mask = mask | (edge_type == relation)

    selected_edge_index = edge_index[:, mask]
    selected_edge_type = edge_type[mask]
    selected_edge_weight = edge_weight[mask]

    return (
        selected_edge_index,
        selected_edge_type,
        selected_edge_weight,
    )


def graph_statistics(edge_index, edge_type):
    """
    Print basic graph statistics.
    """

    print("\nGraph statistics")
    print("----------------")

    print(
        "Number of edges:",
        edge_index.shape[1]
    )

    unique_relations, counts = torch.unique(
        edge_type,
        return_counts=True
    )

    print("\nRelations:")

    for relation, count in zip(
        unique_relations.tolist(),
        counts.tolist()
    ):
        print(
            f"Relation {relation}: {count}"
        )
