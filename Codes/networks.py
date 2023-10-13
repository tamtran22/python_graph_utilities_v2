import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Tuple, Optional, Union
from torch_geometric.typing import OptTensor, Tensor
import torch_scatter

class MeshGraphNet(gnn.MessagePassing):
    r"""MeshGraphNet-based
    params:
        node_in : (size of input node feature, flag to use node encoder)
        node_out : (size of output node feature, flag to use node decoder)
        node_in : (size of input edge feature, flag to use edge encoder)
        node_in : (size of output edge feature, flag to use edge decoder)
    """
    def __init__(self,
        node_in : Tuple[int, bool] = (0, False),
        node_out : Tuple[int, bool] = (0, False), 
        edge_in : Tuple[int, bool] = (0, False),
        edge_out : Tuple[int, bool] = (0, False),
        hidden_size : int = 128,
        n_hiddens : int = 10,
        aggregation : str = 'sum'
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_hiddens = n_hiddens
        self.aggregation = aggregation

        # Node encoder ################################
        if not node_in[1]:
            self.node_encoder = None
            _node_encoder_out_size = node_in[0]
        else:
            self.node_in_size = node_in[0]
            self.node_encoder = nn.Sequential(
                nn.Linear(self.node_in_size, self.hidden_size),
                nn.ReLU()
            )
            _node_encoder_out_size = self.hidden_size

        # Node decoder #################################
        # node mlp
        if not node_out[1]:
            self.node_decoder = None
        else:
            self.node_out_size = node_out[0]
            self.node_decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.node_out_size)
            )
        
        # Edge encoder #################################
        if not edge_in[1]:
            self.edge_encoder = None
            _edge_encoder_out_size = edge_in[0]
        else:
            self.edge_in_size = edge_in[0]
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.node_in_size, self.hidden_size),
                nn.ReLU()
            )
            _edge_encoder_out_size = self.hidden_size
        
        # Edge decoder #################################
        if not edge_out[1]:
            self.edge_decoder = None
        else:
            self.edge_out_size = edge_out[0]
            self.edge_decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.edge_out_size)
            )

        # Edge mlp #####################################
        layer_list = [nn.Linear(2*_node_encoder_out_size+_edge_encoder_out_size, self.hidden_size)]
        for _ in range(self.n_hiddens - 1):
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.edge_mlp = nn.Sequential(*layer_list)
        
    def forward(self, x, edge_index, edge_attr : OptTensor = None, size = None):
        
        if self.node_encoder is not None:
            x = self.node_encoder(x)
        
        if (self.edge_encoder is not None) and (edge_attr is not None):
            edge_attr = self.edge_encoder(edge_attr)
        
        updated_nodes, updated_edges = self.propagate(
            edge_index,
            x = x,
            edge_attr = edge_attr,
            size = size
        )

        if self.node_decoder is not None:
            updated_nodes = self.node_decoder(updated_nodes)

        if self.edge_decoder is not None:
            updated_edges = self.edge_decoder(updated_edges)
        
        return updated_nodes, updated_edges
    
    def message(self, x_i, x_j, edge_attr : OptTensor = None):
        updated_edges = torch.cat([x_i, x_j], dim=1)
        
        if edge_attr is not None:
            updated_edges = torch.cat([updated_edges, edge_attr], dim=1)

        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index, dim_size = None):
        node_dim = 0
        updated_nodes = torch_scatter.scatter(updated_edges, edge_index[0,:],
                                            dim=node_dim, reduce = self.aggregation)
        return updated_nodes, updated_edges






class PARC_reduced(nn.Module):
    def __init__(self,
        n_fields,
        n_timesteps,
        n_hiddenfields,
        n_meshfields,
        n_bcfields,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_fields = n_fields
        self.n_timesteps = n_timesteps
        self.n_hiddenfields = n_hiddenfields
        self.n_meshfields = n_meshfields
        self.n_bcfields = n_bcfields

        self.derivative_solver = MeshGraphNet(
            node_in=(self.n_fields + self.n_hiddenfields + self.n_bcfields, True),
            node_out=(self.n_fields, True),
            hidden_size=2*self.n_hiddenfields,
            n_hiddens=10
        )

        self.mesh_descriptor = MeshGraphNet(
            node_in=(self.n_meshfields, False),
            node_out=(self.n_hiddenfields, True),
            hidden_size=2*self.n_hiddenfields,
            n_hiddens=10
        )
    
    def forward(self,
        F_initial : Tensor,
        mesh_features : Tensor,
        edge_index : Tensor,
        F_boundary : OptTensor = None,
        timesteps : float = None
    ):
        # Encode mesh features
        mesh_features, _ = self.mesh_descriptor(mesh_features, edge_index)

        F_previous = F_initial
        # F_dot_previous 

        F_dots, Fs = [], []
        PINN_errors = []

        # Recurrent formulation
        for timestep in range(1, self.n_timesteps):
            Q_previous = F_previous[:, 1]

            F_temp = torch.cat([mesh_features, F_previous], dim=1)

            if self.n_bcfields > 0:
                F_temp = torch.cat([F_temp, F_boundary[:, timestep].unsqueeze(1)], dim=1)
            
            F_dot, _ = self.derivative_solver(F_temp, edge_index)

            # Numerical scheme
            F_current = F_previous + timesteps * F_dot

            # Save current state
            F_dots.append(F_dot.unsqueeze(1))
            Fs.append(F_current.unsqueeze(1))
            F_previous = F_current.detach()
            # F_dot_previous

            # PINN error
            # P_current = F_current[:, 0]
            # Q_current = F_current[:, 1]
            # PINN_error = P_current + (Uns / timesteps + Vis) * Q_current + Kin * Q_previous \
            #         + (Uns / timesteps) * Q_previous
            # PINN_errors.append(PINN_error.unsqueeze(1))

            # F_current = self.message(F_current)

        F_dots = torch.cat(F_dots, dim=1)
        Fs = torch.cat(Fs, dim=1)
        # PINN_errors = torch.cat(PINN_errors, dim=1)

        return Fs, F_dots
        # return Fs, F_dots, PINN_errors