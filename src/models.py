
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from torch_geometric.nn import RGCNConv


class GCNBaseline(nn.Module):
    """
    Standard two-layer GCN baseline.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_classes=2,
        dropout=0.30
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index
    ):
        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = nn.functional.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        output = self.classifier(x)

        return output


class RGCNBaseline(nn.Module):
    """
    Multi-relational graph convolution baseline.

    This model uses the seven MGTAB relation types.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_relations=7,
        num_classes=2,
        dropout=0.30
    ):
        super().__init__()

        self.conv1 = RGCNConv(
            input_dim,
            hidden_dim,
            num_relations
        )

        self.conv2 = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
        edge_type
    ):
        x = self.conv1(
            x,
            edge_index,
            edge_type
        )

        x = torch.relu(x)

        x = nn.functional.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index,
            edge_type
        )

        x = torch.relu(x)

        output = self.classifier(x)

        return output


class MultiRelationalAttentionModel(nn.Module):
    """
    Proposed-model placeholder based on
    multi-relational graph representations
    and node-level attention.

    Important:
    This is NOT described as a true temporal model
    because the current MGTAB setup does not provide
    graph snapshots for temporal message passing.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_relations=7,
        num_classes=2,
        dropout=0.30
    ):
        super().__init__()

        self.rgcn1 = RGCNConv(
            input_dim,
            hidden_dim,
            num_relations
        )

        self.rgcn2 = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations
        )

        self.attention = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(
        self,
        x,
        edge_index,
        edge_type
    ):
        x = self.rgcn1(
            x,
            edge_index,
            edge_type
        )

        x = torch.relu(x)

        x = self.rgcn2(
            x,
            edge_index,
            edge_type
        )

        x = torch.relu(x)

        attention_score = self.attention(x)

        attention_weight = torch.sigmoid(
            attention_score
        )

        representation = (
            x * attention_weight
        )

        output = self.classifier(
            representation
        )

        return output
