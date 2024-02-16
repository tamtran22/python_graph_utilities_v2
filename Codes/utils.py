from typing import Dict, List
import torch
from torch import Tensor
import torch_geometric.nn as gnn
from torch_geometric.typing import Adj
import torch_scatter
from typing import Callable

class NodeToEdgeLayer(gnn.MessagePassing):
    def __init__(self,
        message: Callable = lambda a, b : b,
        **kwargs
    ):
        self._message = message
        super().__init__(**kwargs)
    
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.propagate(edge_index=edge_index, x=x)
    
    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self._message(x_i, x_j)
    
    def aggregate(self, inputs: Tensor, index: Tensor = None) -> Tensor:
        return inputs

class EdgeToNodeLayer(gnn.MessagePassing):
    def __init__(self, 
        aggregation='sum', 
        **kwargs
    ):
        self.aggregation = aggregation
        super().__init__(**kwargs)
    
    def forward(self, edge_attr: Tensor, edge_index: Adj) -> Tensor:
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr)
    
    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr
    
    def aggregate(self, inputs: Tensor, edge_index: Tensor) -> Tensor:
        node_dim=0
        return torch_scatter.scatter(inputs, edge_index[1,:], dim=node_dim,
                                reduce=self.aggregation)