import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Tuple, Optional, Union, List, Callable
from torch_geometric.typing import OptTensor, Tensor
import torch_scatter



#####################################################################################
######################################################################################
#####################################################################################
## Original multi-processor-layers meshgraphnet
####################################################################################
#####################################################################################
#####################################################################################


# Build a multi-layers perceptron with custom layer sizes and activations
def build_mlp(
    input_size: int,
    hidden_layer_sizes: List[int],
    output_size: int,
    output_activation: nn.Module = nn.Identity(),
    activation: nn.Module = nn.ReLU()
) -> nn.Module:
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes = layer_sizes + [output_size]
    
    n_layers = len(layer_sizes) - 1

    layer_activations = [activation for i in range(n_layers - 1)]
    layer_activations = layer_activations + [output_activation]

    # mlp = nn.Sequential()
    # for i in range(n_layers - 1):
    #     mlp.add_module('nn-'+str(i), nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    #     mlp.add_module('act-'+str(i), layer_activations[i])
    # mlp.add_module('nn-'+str(n_layers-1), nn.Linear(layer_sizes[n_layers-1], layer_sizes[n_layers]))

    mlp = []
    for i in range(n_layers):
        mlp.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        mlp.append(layer_activations[i])
    
    return nn.Sequential(*mlp)


# processor for node and edge features
class ProcessorLayer(gnn.MessagePassing):
    def __init__(self,
        node_input_size: int,
        node_output_size: int,
        edge_input_size: int,
        edge_output_size: int,
        hidden_size: int,
        n_hidden: int
    ):
        super().__init__(aggr='add')

        self.node_fn = nn.Sequential(*[
            build_mlp(input_size=edge_output_size,
                    hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
                    output_size=node_output_size),
            nn.LayerNorm(node_output_size)
        ])

        self.edge_fn = nn.Sequential(*[
            build_mlp(input_size=2*node_input_size+edge_input_size,
                    hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
                    output_size=edge_output_size),
            nn.LayerNorm(edge_output_size)
        ])
    
    def forward(self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: torch.tensor
    ):
        x_residual = x
        edge_attr_residual = edge_attr

        x, edge_attr = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr
        )

        x = x + x_residual
        edge_attr = edge_attr + edge_attr_residual

        return x, edge_attr
    
    def message(self,
        x_i: torch.tensor,
        x_j: torch.tensor,
        edge_attr: torch.tensor
    ) -> torch.tensor:
        edge_attr = torch.cat([x_i, x_j, edge_attr], dim=-1)
        edge_attr = self.edge_fn(edge_attr)

        return edge_attr
    
    # def update(self,
    #     x_updated: torch.tensor,
    #     x: torch.tensor,
    #     edge_attr:  torch.tensor         
    # ):
    #     x_updated = torch.cat([x_updated, x], dim=-1)
    #     x_updated = self.node_fn(x_updated)
    #     return x_updated, edge_attr
    def aggregate(self, edge_attr, edge_index, dim_size=None):
        node_dim=0
        x = torch_scatter.scatter(edge_attr, edge_index[1,:], dim=node_dim,
                                reduce='sum')
        x = self.node_fn(x)
        return x, edge_attr


# processor networks
class Processor(gnn.MessagePassing):
    def __init__(self,
        node_input_size: int,
        node_output_size: int,
        edge_input_size: int,
        edge_output_size: int,
        n_message_passing_steps: int,
        n_hidden: int,
        hidden_size: int,
    ):
        super().__init__(aggr='sum')
        self.processor_layers = nn.Sequential(*[
            ProcessorLayer(
                node_input_size,
                node_output_size,
                edge_input_size,
                edge_output_size,
                hidden_size,
                n_hidden
            )
        for _ in range(n_message_passing_steps)])
    
    def forward(self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: torch.tensor
    ):
        for processor_layer in self.processor_layers:
            x, edge_attr = processor_layer(x, edge_index, edge_attr)
        return x, edge_attr


# encoder
class Encoder(nn.Module):
    def __init__(self,
        node_input_size: int,
        node_output_size: int,
        edge_input_size: int,
        edge_output_size: int,
        n_hidden: int,
        hidden_size: int,
    ):
        super().__init__()
        
        self.node_fn = nn.Sequential(*[
            build_mlp(input_size=node_input_size,
                    hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
                    output_size=node_output_size),
            nn.LayerNorm(node_output_size)
        ])

        self.edge_fn = nn.Sequential(*[
            build_mlp(input_size=edge_input_size,
                    hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
                    output_size=edge_output_size),
            nn.LayerNorm(edge_output_size)
        ])
    
    def forward(self,
        x: torch.tensor,
        edge_attr: torch.tensor
    ):
        return self.node_fn(x), self.edge_fn(edge_attr)


# decoder
class Decoder(nn.Module):
    def __init__(self,
        node_input_size: int,
        node_output_size: int,
        # edge_input_size: int,
        # edge_output_size: int,
        n_hidden: int,
        hidden_size: int,
    ):
        super().__init__()
        
        self.node_fn = nn.Sequential(*[
            build_mlp(input_size=node_input_size,
                    hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
                    output_size=node_output_size),
            # nn.LayerNorm(node_output_size)
        ])

        # self.edge_fn = nn.Sequential(*[
        #     build_mlp(input_size=edge_input_size,
        #             hidden_layer_sizes=[hidden_size for _ in range(n_hidden)],
        #             output_size=edge_output_size),
        #     nn.LayerNorm(edge_output_size)
        # ])
    
    def forward(self,
        x: torch.tensor,
        # edge_attr: torch.tensor
    ):
        return self.node_fn(x) #, self.edge_fn(edge_attr)


# Meshgraphnet
class MeshGraphNet(nn.Module):
    def __init__(self,
        node_input_size: int,
        node_output_size: int,
        edge_input_size: int,
        edge_output_size: int,
        n_hidden: int,
        hidden_size: int,
        n_latent: int,
        latent_size: int
    ):
        super().__init__()
        self.encoder = Encoder(
            node_input_size=node_input_size,
            node_output_size=latent_size,
            edge_input_size=edge_input_size,
            edge_output_size=latent_size,
            n_hidden=n_hidden,
            hidden_size=hidden_size
        )
        self.processor = Processor(
            node_input_size=latent_size,
            node_output_size=latent_size,
            edge_input_size=latent_size,
            edge_output_size=latent_size,
            n_message_passing_steps=n_latent,
            n_hidden=4, #test thin processor layer
            hidden_size=hidden_size
        )
        self.decoder = Decoder(
            node_input_size=latent_size,
            node_output_size=node_output_size,
            n_hidden=n_hidden,
            hidden_size=hidden_size
        )
    
    def forward(self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: torch.tensor  
    ):
        x, edge_attr = self.encoder(x, edge_attr)
        x, edge_attr = self.processor(x, edge_index, edge_attr)
        x = self.decoder(x)
        return x


# # Recurrent formulation network
# def recurrent_formulation(
#     model,
#     data,
#     integration: Callable = None,
#     forward_sequence: bool = True
# ):
#     # Define time tensor
#     time = torch.zeros(data.pressure.size())
#     timestep = 4.8 / 200
#     for i in range(time.size(1)):
#         time[:,i] = i * timestep
#     n_time = time.size(1)

#     # recurrent loop
#     F = []
#     F_current = torch.cat([
#         data.pressure[:,0].unsqueeze(1), 
#         data.flowrate[:,0].unsqueeze(1)
#     ], dim=1)
#     for i in range(n_time):
#         if forward_sequence:
#             x = torch.cat([
#                 data.pressure[:,i].unsqueeze(1), 
#                 data.flowrate[:,i].unsqueeze(1),
#                 data.node_attr,
#                 time[:,i].unsqueeze(1)
#             ], dim=1).float()
#         else:
#             x = torch.cat([
#                 F_current.detach(),
#                 data.node_attr,
#                 time[:,i].unsqueeze(1)
#             ], dim=1).float()
#         edge_index = data.edge_index
#         edge_attr = data.edge_attr.float()

#         F_current = model(x, edge_index, edge_attr)

#         if integration is not None:
#             F_current = integration(F_current)

#         F.append(F_current.unsqueeze(1))
    
#     return torch.cat(F, dim=1)



class RecurrentFormulationNetwork(nn.Module):
    def __init__(self,
        n_field: int,
        n_meshfield: Tuple[int, int],
        n_boundaryfield: int,
        n_hidden: int,
        hidden_size: int,
        n_latent: int,
        latent_size: int,
        integration: Callable = None           
    ):
        super().__init__()
        self.mgn = MeshGraphNet(
            node_input_size=n_field+n_meshfield[0]+n_boundaryfield+1,
            node_output_size=n_field,
            edge_input_size=n_meshfield[1],
            edge_output_size=0,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            n_latent=n_latent,
            latent_size=latent_size
        )
        self.integration = integration

    def forward(self,
        F,
        edge_index,
        meshfield,
        time,
        boundaryfield,
        forward_sequence=False
    ):
        timestep = 4.8 / 200
        # initial F
        if not forward_sequence:
            F_current = F
        
        # recurrent loop
        Fs = []
        for i in range(time.size(1) - 1):
            # prepare time
            time_current = time[:,i].unsqueeze(1)
            # prepare boundary condition
            # if boundaryfield is not None:
            #     boundaryfield_current = boundaryfield[:,i]
            # else:
            #     boundaryfield_current = torch.tensor([]).to(self.device)
            # prepare initial condition
            if forward_sequence:
                F_current = F[:,i,:]
            
            x = torch.cat([
                F_current,
                meshfield[0],
                # boundaryfield_current,
                time_current
            ], dim=1)
            
            F_dot_current = self.mgn(x=x, edge_index=edge_index,
                            edge_attr=meshfield[1])
            
            # if self.integration is not None:
            #     F_current = self.integration(F_current)
            F_next = F_current + timestep*F_dot_current

            Fs.append(F_next.unsqueeze(1)) 
            F_current = F_next.detach()
        
        return torch.cat(Fs, dim=1)


