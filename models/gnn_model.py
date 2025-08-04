from .base_model import BaseModel
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None, **kwargs):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x
