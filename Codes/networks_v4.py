import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Union, Callable, List, Tuple
from torch_geometric.typing import OptTensor, Tensor
    




##########################
class RecurrentFormulationNetwork(nn.Module):
    def __init__(self,
        n_field: int,
        n_meshfield: Tuple[int, int],
        n_boundaryfield: int,
        # n_globalfield: int,
        n_hidden: int,
        hidden_size: int,
        integration: Callable = None   ,
        **kwargs 
    ) -> None:
        super().__init__(**kwargs)
        self.integration = integration
        self.pre_net = gnn.Sequential('x, edge_index',
            [
                (gnn.GCNConv(in_channels=n_field + n_meshfield[0] + n_boundaryfield, out_channels=hidden_size), 
                        'x, edge_index -> x'),
                (nn.LayerNorm(hidden_size), 'x -> x'),
                nn.Mish(inplace=True),
                (gnn.GCNConv(in_channels=hidden_size, out_channels=hidden_size),
                        'x, edge_index -> x'),
                nn.Mish(inplace=True)
            ]
        )
        self.net = gnn.GraphUNet(
            in_channels=hidden_size, # + n_globalfield,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            depth=n_hidden,
            pool_ratios=0.5,
            sum_res=True,
            act=torch.nn.functional.mish
        )
        self.post_net = gnn.Sequential('x, edge_index',
            [
                (gnn.GCNConv(in_channels=hidden_size, out_channels=hidden_size), 
                        'x, edge_index -> x'),
                nn.Mish(inplace=True),
                (gnn.GCNConv(in_channels=hidden_size, out_channels=hidden_size),
                        'x, edge_index -> x'),
                nn.Mish(inplace=True),
                (nn.Linear(in_features=hidden_size, out_features=hidden_size),
                        'x -> x'),
                nn.Mish(),
                (nn.Linear(in_features=hidden_size, out_features=hidden_size),
                        'x -> x'),
                nn.Mish(),
                (nn.Linear(in_features=hidden_size, out_features=n_field),
                        'x -> x')
            ]
        )
    
    def reset_parameters(self):
        pass

    def forward(self,
        F: torch.tensor,
        edge_index: torch.tensor,
        meshfield: Tuple[torch.tensor, torch.tensor],
        boundaryfield: torch.tensor=None,
        # globalfield: torch.tensor=None,
        forward_sequence=False,
        n_time: int=1
    ):
        timestep = 4.8 / 200

        # meshfield = self.mesh_descriptor(
        #     x=meshfield[0],
        #     edge_index=edge_index,
        #     edge_attr=meshfield[1]
        # )
        
        if not forward_sequence:
            F_current = F

        Fs = []

        for i in range(n_time - 1):
            boundary_current = boundaryfield[:,i].unsqueeze(1)
            if forward_sequence:
                F_current = F[:,i,:]

            x = torch.cat([
                F_current,
                boundary_current,
                meshfield[0]
            ], dim=1)

            # print('prenet')
            x = self.pre_net(x=x, edge_index=edge_index)

            # print('net')
            x = self.net(x=x, edge_index = edge_index)

            # print('postnet')
            F_dot_current = self.post_net(x=x, edge_index=edge_index)

            F_next = F_current + timestep*F_dot_current
            
            Fs.append(F_next.unsqueeze(1))
            F_current = F_next.detach()

        return torch.cat(Fs, dim=1)