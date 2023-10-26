import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Tuple, Optional, Union
from torch_geometric.typing import OptTensor, Tensor
from loss import LossFunction
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
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
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
                nn.Linear(self.edge_in_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
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
        for i in range(self.n_hiddens - 1):
            layer_list.append(nn.ReLU())
            # if i == 2:
            #     layer_list.append(nn.LayerNorm(self.hidden_size))
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
            # updated_edges = edge_attr + self.edge_mlp(updated_edges) # skip connection
            updated_edges = self.edge_mlp(updated_edges)
        else:
            updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index, dim_size = None):
        node_dim = 0
        updated_nodes = torch_scatter.scatter(updated_edges, edge_index[1,:],
                                            dim=node_dim, reduce = self.aggregation)
        return updated_nodes, updated_edges



class LSTMMeshGraphNet(gnn.MessagePassing):
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
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
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
                nn.Linear(self.edge_in_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
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
        self.edge_mlp = nn.LSTM(
            input_size=2*_node_encoder_out_size+_edge_encoder_out_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_hiddens,
            dropout=0.1
        )
        
    def forward(self, x, edge_index, edge_attr : OptTensor = None,
                hx : Optional[Tuple[Tensor,Tensor]] = None, size = None):

        if self.node_encoder is not None:
            x = self.node_encoder(x)

        if (self.edge_encoder is not None) and (edge_attr is not None):
            edge_attr = self.edge_encoder(edge_attr)
        
        # if hidden_edge_features is None:
        #     hidden_edge_features = torch.zeros((edge_attr.size(0), self.hidden_size)).float()
        #     cell_edge_features = torch.zeros((edge_attr.size(0), self.hidden_size)).float()
        # print(hidden_edge_features.size(), cell_edge_features.size())

        updated_nodes, updated_edges, hx0, hx1 = self.propagate(
            edge_index,
            x = x,
            edge_attr = edge_attr,
            hx = hx,
            size = size
        )

        if self.node_decoder is not None:
            updated_nodes = self.node_decoder(updated_nodes)

        if self.edge_decoder is not None:
            updated_edges = self.edge_decoder(updated_edges)
        
        return updated_nodes, updated_edges, (hx0.detach(), hx1.detach())
    
    def message(self, x_i, x_j, edge_attr : OptTensor = None, hx : OptTensor=None):
        updated_edges = torch.cat([x_i, x_j], dim=1)
        if edge_attr is not None:
            updated_edges = torch.cat([updated_edges, edge_attr], dim=1)
            # updated_edges = edge_attr + self.edge_mlp(updated_edges) # skip connection
            updated_edges, hx = self.edge_mlp(
                input=updated_edges, hx=hx)
        else:
            updated_edges, hx = self.edge_mlp(
                input=updated_edges, hx=hx)
        return updated_edges, hx[0], hx[1]
    
    def aggregate(self, updated_edges, edge_index, dim_size = None):
        node_dim = 0
        updated_edges, hx0, hx1 = updated_edges
        updated_nodes = torch_scatter.scatter(updated_edges, edge_index[1,:],
                                            dim=node_dim, reduce = self.aggregation)
        return updated_nodes, updated_edges, hx0, hx1
    






class PARC(gnn.MessagePassing):
    def __init__(self,
        n_fields,
        n_timesteps,
        n_meshfields, # Tuple(n_node_fields, n_mesh_fields)
        n_bcfields,
        n_hiddenfields,
        n_hiddens,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_fields = n_fields
        self.n_timesteps = n_timesteps
        self.n_hiddenfields = n_hiddenfields
        self.n_meshfields = n_meshfields
        self.n_bcfields = n_bcfields
        self.n_hiddens = n_hiddens

        self.derivative_solver = LSTMMeshGraphNet(
            node_in=(self.n_fields + self.n_bcfields, True),
            edge_in=(self.n_hiddenfields, False),
            node_out=(self.n_fields, True),
            hidden_size=self.n_hiddenfields,
            n_hiddens=self.n_hiddens
        )

        self.mesh_descriptor = MeshGraphNet(
            node_in=(self.n_meshfields[0], True),
            edge_in=(self.n_meshfields[1], True),
            edge_out=(self.n_hiddenfields, True),
            hidden_size=self.n_hiddenfields,
            n_hiddens=self.n_hiddens
        )

        self.msg_net = MessageNet()
    
    def forward(self,
        F_initial : Tensor,
        mesh_features : Tuple[Tensor, Tensor],
        edge_index : Tensor,
        F_boundary : OptTensor = None,
        timestep : float = None
    ):
        # Encode mesh features
        _, _mesh_features = self.mesh_descriptor(
            x = mesh_features[0], 
            edge_index=edge_index,
            edge_attr = mesh_features[1]
        )       

        F_previous = F_initial
        # F_dot_previous = torch.zeros_like(F_initial).float()
        F_hidden = None

        F_dots, Fs = [], []

        # Recurrent formulation
        for timestep in range(1, self.n_timesteps):
            F_temp = F_previous

            if self.n_bcfields > 0:
                F_temp = torch.cat([F_temp, F_boundary[:, timestep].unsqueeze(1)], dim=1)
            
            F_dot, _, F_hidden = self.derivative_solver(
                x = F_temp, 
                edge_index=edge_index,
                edge_attr = _mesh_features,
                hx = F_hidden
            )

            # Numerical scheme
            F_current = F_previous + timestep * F_dot
            # F_current = F_previous + timestep * (F_dot + F_dot_previous)

            # Save current state
            F_dots.append(F_dot.unsqueeze(1))
            Fs.append(F_current.unsqueeze(1))
            F_previous = F_current.detach()
            F_dot_previous = F_dot.detach()

        F_dots = torch.cat(F_dots, dim=1)
        Fs = torch.cat(Fs, dim=1)

        P = torch.cat([F_initial[:,0].unsqueeze(1), Fs[:,:,0]], dim=1)
        Q = torch.cat([F_initial[:,1].unsqueeze(1), Fs[:,:,1]], dim=1)
        # P, Q = self.propagate(
        #     edge_index,
        #     P = P,
        #     Q = Q
        # )
        P = self.msg_net(P, edge_index)
        Q = self.msg_net(Q, edge_index)

        # PINN_loss = P[:,1:] + Q[:,1:] + Q[:,:-1]
        return Fs, F_dots
        # return Fs, F_dots, PINN_errors
    
    # def message(self, P_i, P_j, Q_i, Q_j):
    #     updated_P = P_i - P_j
    #     updated_Q = Q_i - Q_j
    #     return updated_P, updated_Q
    
    # def aggregate(self, updated_edges, edge_index, dim_size = None):
    #     return updated_edges[0], updated_edges[1]


class MessageNet(gnn.MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input, edge_index):
        return self.propagate(
            edge_index,
            input = input
        )

    def message(self, input_i, input_j):
        return input_i - input_j
    
    def aggregate(self, updated_edges, edge_index, dim_size = None):
        return updated_edges