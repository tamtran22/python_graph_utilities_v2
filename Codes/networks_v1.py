import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Union, Callable, List, Tuple
from torch_geometric.typing import OptTensor, Tensor
import torch_scatter

###############################################################################
# Build a multi-layers perceptron with custom layer sizes and activations
def build_mlp(
    input_size: int,
    hidden_layer_sizes: List[int],
    output_size: int,
    output_activation: nn.Module = None,
    output_norm: bool = False,
    activation: nn.Module = nn.ReLU()
) -> nn.Module:
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes = layer_sizes + [output_size]
    
    n_layers = len(layer_sizes) - 1

    layer_activations = [activation for i in range(n_layers - 1)]
    layer_activations = layer_activations + [output_activation]

    # mlp = nn.Sequential()
    mlp = []
    for i in range(n_layers):
        # mlp.add_module('nn-'+str(i), nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        # mlp.add_module('act-'+str(i), layer_activations[i])
        mlp.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if layer_activations[i] is not None:
            mlp.append(layer_activations[i])   
    if output_norm:
        # mlp.add_module('norm', nn.LayerNorm(layer_sizes[-1]))
        mlp.append(nn.LayerNorm(layer_sizes[-1]))
    return nn.Sequential(*mlp)

###############################################################################
# processor for node and edge features
class ProcessorLayer(gnn.MessagePassing):
    def __init__(self,
        node_input_size: int,
        edge_input_size: int,
        node_output_size: int,
        edge_output_size: int,
        hidden_size: int,
        n_hidden: int,
        aggregation: str = 'sum'
    ):
        super().__init__(aggr='add')
        self.aggregation = aggregation

        self.node_fn = build_mlp(input_size=node_input_size+hidden_size,
                    hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
                    output_size=node_output_size)

        self.edge_mlp = build_mlp(input_size=2*node_input_size+edge_input_size,
                    hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
                    output_size=hidden_size)
        
        self.edge_fn = build_mlp(input_size=hidden_size,
                    hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
                    output_size=edge_output_size)
    
    def forward(self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: OptTensor = None,
        size=None
    ):
        # x_residual = x
        # edge_attr_residual = edge_attr

        out, updated_edges = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            size=size
        )
        

        updated_nodes = torch.cat([x,out], dim=1) # ?

        updated_nodes = self.node_fn(updated_nodes)
        updated_edges = self.edge_fn(updated_edges)

        return updated_nodes, updated_edges
    
    def message(self,
        x_i: torch.tensor,
        x_j: torch.tensor,
        edge_attr: OptTensor = None
    ) -> torch.tensor:
        updated_edges = torch.cat([x_i, x_j], dim=-1)
        if edge_attr is not None:
            updated_edges = torch.cat([updated_edges, edge_attr], dim=-1)

        updated_edges = self.edge_mlp(updated_edges)

        return updated_edges
    
    def aggregate(self, updated_edges, edge_index, dim_size=None):
        node_dim=0
        out = torch_scatter.scatter(updated_edges, edge_index[0,:], dim=node_dim,
                                reduce=self.aggregation)
        return out, updated_edges


############################################################################
# Mesh graph net
class MeshGraphNet(nn.Module):
    def __init__(self,
        node_input_size: Tuple[int, bool],
        node_output_size: Tuple[int, bool],
        edge_input_size: Tuple[int, bool],
        edge_output_size: Tuple[int, bool],
        n_hidden: int,
        hidden_size: int,
        n_latent: int,
        latent_size: int,
        n_hidden_per_processor: int = 4,
        activation: nn.Module = nn.Mish()
    ):
        super().__init__()

        # Node encoder #######################################
        if (node_input_size[0] <= 0) or (not node_input_size[1]):
            self.node_encoder = None
            latent_node_input_size = node_input_size[0]
        else:
            self.node_encoder = build_mlp(
                input_size=node_input_size[0],
                hidden_layer_sizes=[hidden_size]*n_hidden,
                output_size=latent_size,
                output_activation=None,
                output_norm=False,
                activation=activation
            )
            latent_node_input_size = latent_size

        # Edge encoder #######################################
        if (edge_input_size[0] <= 0) or (not edge_input_size[1]):
            self.edge_encoder = None
            latent_edge_input_size = edge_input_size[0]
        else:
            self.edge_encoder = build_mlp(
                input_size=edge_input_size[0],
                hidden_layer_sizes=[hidden_size]*n_hidden,
                output_size=latent_size,
                output_activation=None,
                output_norm=False,
                activation=activation
            )
            latent_edge_input_size = latent_size

        # Node decoder #######################################
        if (node_output_size[0] <= 0) or (not node_output_size[1]):
            self.node_decoder = None
            latent_node_output_size = node_output_size[0]
        else:
            self.node_decoder = build_mlp(
                input_size=latent_size,
                hidden_layer_sizes=[hidden_size]*n_hidden,
                output_size=node_output_size[0],
                output_activation=None,
                output_norm=False,
                activation=activation
            )
            latent_node_output_size = latent_size

        # Edge decoder #######################################
        if (edge_output_size[0] <= 0) or (not edge_output_size[1]):
            self.edge_decoder = None
            latent_edge_output_size = edge_output_size[0]
        else:
            self.edge_decoder = build_mlp(
                input_size=latent_size,
                hidden_layer_sizes=[hidden_size]*n_hidden,
                output_size=edge_output_size[0],
                output_activation=None,
                output_norm=False,
                activation=activation
            )
            latent_edge_output_size = latent_size

        # Processor layers ###################################
        self.processor = nn.ModuleList()
        self.processor.append(ProcessorLayer(
            node_input_size=latent_node_input_size,
            edge_input_size=latent_edge_input_size,
            node_output_size=latent_size,
            edge_output_size=latent_size,
            hidden_size=hidden_size,
            n_hidden=n_hidden_per_processor
        ))
        for _ in range(1, n_latent - 1):
            # Second layers onward input edge attr from the previous layer.
            self.processor.append(
                ProcessorLayer(
                    node_input_size=latent_size,
                    edge_input_size=latent_size,
                    node_output_size=latent_size,
                    edge_output_size=latent_size,
                    hidden_size=hidden_size,
                    n_hidden=n_hidden_per_processor
                )
            )
        self.processor.append(ProcessorLayer(
            node_input_size=latent_size,
            edge_input_size=latent_size,
            node_output_size=latent_node_output_size,
            edge_output_size=latent_edge_output_size,
            hidden_size=hidden_size,
            n_hidden=n_hidden_per_processor
        ))
        
        # self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_attr : OptTensor = None):
        # input tuple
        if self.node_encoder is not None:
            x = self.node_encoder(x)
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        for i in range(len(self.processor)):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
        # output tuple
        if self.node_decoder is not None:
            x = self.node_decoder(x)
        if self.edge_decoder is not None:
            edge_attr = self.edge_decoder(edge_attr)
        return x, edge_attr
    


##########################
class RecurrentFormulationNetwork(nn.Module):
    def __init__(self,
        n_field: int,
        n_meshfield: Tuple[int, int],
        n_boundaryfield: int,
        n_hidden: int,
        hidden_size: int,
        n_latent: int,
        latent_size: int,
        integration: Callable = None   ,
        **kwargs 
    ) -> None:
        super().__init__(**kwargs)
        self.integration = integration
        
        self.mesh_descriptor = MeshGraphNet(
            node_input_size=(n_meshfield[0], True),
            edge_input_size=(n_meshfield[1], True),
            node_output_size=(latent_size, False),
            edge_output_size=(0, False), # no edge feature
            n_hidden=n_hidden, # encoder depth
            hidden_size=hidden_size,
            n_latent=n_latent, # processor depth
            latent_size=latent_size,
            n_hidden_per_processor=2
        )

        self.differentiator = MeshGraphNet(
            node_input_size=(n_field+latent_size+n_boundaryfield, False), # time + 1
            edge_input_size=(0, False), # node edge feature
            node_output_size=(n_field, True),
            edge_output_size=(0, False),
            n_hidden=n_hidden, # decoder depth
            hidden_size=hidden_size,
            n_latent=n_latent, # processor depth
            latent_size=latent_size,
            n_hidden_per_processor=2
        )

        # self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self,
        F: torch.tensor,
        edge_index: torch.tensor,
        meshfield: Tuple[torch.tensor, torch.tensor],
        time: torch.tensor,
        boundaryfield: torch.tensor=None,
        forward_sequence=False,
        n_time: int=1
    ):
        timestep = 4.8 / 200

        meshfield = self.mesh_descriptor(
            x=meshfield[0],
            edge_index=edge_index,
            edge_attr=meshfield[1]
        )
        
        # F_previous = F_initial
        # F_dot_previous = torch.zeros_like(F_initial)
        if not forward_sequence:
            F_current = F

        F_dots, Fs = [], []

        for i in range(n_time - 1):
            boundary_current = boundaryfield[:,i].unsqueeze(1)
            if forward_sequence:
                F_current = F[:,i,:]
            x = torch.cat([
                F_current,
                meshfield[0],
                boundary_current
            ], dim=1)
            F_dot_current, _ = self.differentiator(
                x=x,
                edge_index = edge_index,
                edge_attr=None
            )

            F_next = F_current + timestep*F_dot_current
            
            Fs.append(F_next.unsqueeze(1))
            F_current = F_next.detach()

        return torch.cat(Fs, dim=1)