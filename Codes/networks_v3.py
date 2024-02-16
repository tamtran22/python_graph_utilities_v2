import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.nn as gnn
from typing import (
    Optional, Union, Callable, List, Tuple
)
from torch_geometric.typing import (
    Adj, OptPairTensor, OptTensor, SparseTensor, torch_sparse
)
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_attr: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_attr = add_remaining_self_loops(
            edge_index, edge_attr, fill_value, num_nodes)

    if edge_attr is None:
        edge_attr = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_attr, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_attr = deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col]

    return edge_index, edge_attr


################################################
class GCNConvLayer(gnn.MessagePassing):
    r'''Transform node and edge features to node input for GCNConv
    '''
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
        in_channels: Tuple[int,int],
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = False,
        bias = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize
        
        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize


        # self.lin = gnn.Linear(2*in_channels[0]+in_channels[1], out_channels, 
        #                     bias=False, weight_initializer='glorot')
        
        self.lin = gnn.Linear(in_channels[0]+in_channels[1], out_channels, 
                            bias=False, weight_initializer='glorot')

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        gnn.inits.zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None
    
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_attr = gcn_norm(  # yapf: disable
                        edge_index, edge_attr, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_attr)
                else:
                    edge_index, edge_attr = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_attr, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor = None) -> Tensor:
        
        # updated_edges = torch.cat([x_i, x_j], dim=-1)
        updated_edges = x_j
        if edge_attr is not None:
            updated_edges = torch.cat([updated_edges, edge_attr], dim=-1)
        
        updated_edges = self.lin(updated_edges)
        
        return updated_edges
    
    def message_and_aggregate(self, adj_t: SparseTensor | Tensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)






##########################
# class RecurrentFormulationNetwork(nn.Module):
class PARC_Graph(nn.Module):
    def __init__(self,
        n_field: int,
        n_meshfield: Tuple[int, int],
        n_boundaryfield: int,
        hidden_size: int,
        unet_depth: int = 3,
        activation: Callable = F.relu,
        **kwargs 
    ) -> None:
        super().__init__(**kwargs)

        self.activation = activation

        self.mesh_descriptor = gnn.Sequential('x, edge_index, edge_attr',[
            (GCNConvLayer(in_channels=n_meshfield, out_channels=hidden_size), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(inplace=True),
            (gnn.GraphUNet(in_channels=hidden_size, hidden_channels=hidden_size, out_channels=hidden_size, \
                            depth=unet_depth, pool_ratios=0.5, sum_res=True, act=self.activation), \
                            'x, edge_index -> x'),
        ])

        # derivative solver
        self.deriv_res_block1_conv0 = GCNConvLayer(in_channels=(n_field + n_boundaryfield + hidden_size, 0), out_channels=int(hidden_size / 2))
        self.deriv_res_block1_conv1 = GCNConvLayer(in_channels=(int(hidden_size / 2), 0), out_channels=int(hidden_size / 2))
        self.deriv_res_block1_conv2 = GCNConvLayer(in_channels=(int(hidden_size / 2), 0), out_channels=int(hidden_size / 2))
        
        self.deriv_res_block2_conv0 = GCNConvLayer(in_channels=(int(hidden_size / 2), 0), out_channels=hidden_size) # input b1_conv0 (h/2) + b1_conv2 (h/2)
        self.deriv_res_block2_conv1 = GCNConvLayer(in_channels=(hidden_size, 0), out_channels=hidden_size)
        self.deriv_res_block2_conv2 = GCNConvLayer(in_channels=(hidden_size, 0), out_channels=hidden_size)
        
        self.deriv_res_block3_conv0 = GCNConvLayer(in_channels=(hidden_size, 0), out_channels=hidden_size) # input b2_conv0 (h) + b2_conv2 (h)
        self.deriv_res_block3_conv1 = GCNConvLayer(in_channels=(hidden_size, 0), out_channels=int(hidden_size / 2))
        self.deriv_res_block3_conv2 = GCNConvLayer(in_channels=(int(hidden_size / 2), 0), out_channels=int(hidden_size / 4))

        self.deriv_F_dot = GCNConvLayer(in_channels=(int(hidden_size / 4), 0), out_channels=n_field)

        # integral solver
        self.int_res_block1_conv0 = GCNConvLayer(in_channels=(n_field, 0), out_channels=int(hidden_size / 2))
        self.int_res_block1_conv1 = GCNConvLayer(in_channels=(int(hidden_size / 2), 0), out_channels=int(hidden_size / 2))
        self.int_res_block1_conv2 = GCNConvLayer(in_channels=(int(hidden_size / 2), 0), out_channels=int(hidden_size / 2))
        
        self.int_res_block2_conv0 = GCNConvLayer(in_channels=(int(hidden_size / 2), 0), out_channels=hidden_size)
        self.int_res_block2_conv1 = GCNConvLayer(in_channels=(hidden_size, 0), out_channels=hidden_size)
        self.int_res_block2_conv2 = GCNConvLayer(in_channels=(hidden_size, 0), out_channels=hidden_size)
        
        self.int_res_block3_conv0 = GCNConvLayer(in_channels=(hidden_size, 0), out_channels=hidden_size)
        self.int_res_block3_conv1 = GCNConvLayer(in_channels=(hidden_size, 0), out_channels=int(hidden_size / 2))
        self.int_res_block3_conv2 = GCNConvLayer(in_channels=(int(hidden_size / 2), 0), out_channels=int(hidden_size / 4))

        self.int_F = GCNConvLayer(in_channels=(int(hidden_size / 4), 0), out_channels=n_field)

        self.reset_parameters()

    
    def reset_parameters(self):
        self.mesh_descriptor.reset_parameters()

        self.deriv_res_block1_conv0.reset_parameters()
        self.deriv_res_block1_conv1.reset_parameters()
        self.deriv_res_block1_conv2.reset_parameters()
        self.deriv_res_block2_conv0.reset_parameters()
        self.deriv_res_block2_conv1.reset_parameters()
        self.deriv_res_block2_conv2.reset_parameters()
        self.deriv_res_block3_conv0.reset_parameters()
        self.deriv_res_block3_conv1.reset_parameters()
        self.deriv_res_block3_conv2.reset_parameters()
        self.deriv_F_dot.reset_parameters()

        self.int_res_block1_conv0.reset_parameters()
        self.int_res_block1_conv1.reset_parameters()
        self.int_res_block1_conv2.reset_parameters()
        self.int_res_block2_conv0.reset_parameters()
        self.int_res_block2_conv1.reset_parameters()
        self.int_res_block2_conv2.reset_parameters()
        self.int_res_block3_conv0.reset_parameters()
        self.int_res_block3_conv1.reset_parameters()
        self.int_res_block3_conv2.reset_parameters()
        self.int_F.reset_parameters()
        

    def derivative_solver(self, F_input: Tensor, edge_index: Adj, meshfield_encoded: Tensor) -> Tensor:
        concat = torch.cat([F_input, meshfield_encoded], dim=-1)

        b1_conv0 = self.activation(self.deriv_res_block1_conv0(concat, edge_index))
        b1_conv1 = self.activation(self.deriv_res_block1_conv1(b1_conv0, edge_index))
        b1_conv2 = self.deriv_res_block1_conv2(b1_conv1, edge_index)
        b1_add = self.activation(b1_conv0 + b1_conv2)

        b2_conv0 = self.activation(self.deriv_res_block2_conv0(b1_add, edge_index))
        b2_conv1 = self.activation(self.deriv_res_block2_conv1(b2_conv0, edge_index))
        b2_conv2 = self.deriv_res_block2_conv2(b2_conv1, edge_index)
        b2_add = self.activation(b2_conv0 + b2_conv2)

        b3_conv0 = self.activation(self.deriv_res_block3_conv0(b2_add, edge_index))
        b3_conv1 = self.activation(self.deriv_res_block3_conv1(b3_conv0, edge_index))
        b3_conv2 = self.activation(self.deriv_res_block3_conv2(b3_conv1, edge_index))
        b3_add = F.dropout(b3_conv2, p=0.2)

        F_dot = self.deriv_F_dot(b3_add, edge_index)
        F_dot = F.tanh(F_dot)
        return F_dot
    
    def integral_solver(self, F_dot: Tensor, edge_index: Adj) -> Tensor:

        b1_conv0 = self.activation(self.int_res_block1_conv0(F_dot, edge_index))
        b1_conv1 = self.activation(self.int_res_block1_conv1(b1_conv0, edge_index))
        b1_conv2 = self.int_res_block1_conv2(b1_conv1, edge_index)
        b1_add = self.activation(b1_conv0 + b1_conv2)

        b2_conv0 = self.activation(self.int_res_block2_conv0(b1_add, edge_index))
        b2_conv1 = self.activation(self.int_res_block2_conv1(b2_conv0, edge_index))
        b2_conv2 = self.int_res_block2_conv2(b2_conv1, edge_index)
        b2_add = self.activation(b2_conv0 + b2_conv2)
        b2_add = F.dropout(b2_add, p=0.2)

        b3_conv0 = self.activation(self.int_res_block3_conv0(b2_add, edge_index))
        b3_conv1 = self.activation(self.int_res_block3_conv1(b3_conv0, edge_index))
        b3_conv2 = self.activation(self.int_res_block3_conv2(b3_conv1, edge_index))
        b3_add = F.dropout(b3_conv2, p=0.2)

        F_int = self.int_F(b3_add, edge_index)
        return F_int


    def forward(self,
        F_input: torch.tensor,
        edge_index: torch.tensor,
        meshfield: Tuple[torch.tensor, torch.tensor],
        boundaryfield: torch.tensor=None,
        forward_sequence=False,
        n_time: int=1
    ):

        meshfield_encoded = self.mesh_descriptor(
            x=meshfield[0],
            edge_index=edge_index,
            edge_attr=meshfield[1]
        )
        
        if not forward_sequence:
            F_current = F_input

        Fs, F_dots = [], []

        for i in range(n_time - 1):

            if forward_sequence:
                F_current = F_input[:,i,:]
            

            if boundaryfield is not None:
                F_current = torch.cat([F_current, boundaryfield[:,i].unsqueeze(1)], dim=1)

            F_dot_current = self.derivative_solver(F_input=F_current, edge_index=edge_index,
                                                   meshfield_encoded=meshfield_encoded)
            
            F_next = self.integral_solver(F_dot=F_dot_current, edge_index=edge_index)
            
            Fs.append(F_next.unsqueeze(1))
            F_dots.append(F_dot_current.unsqueeze(1))
            F_current = F_next.detach()

        return torch.cat(Fs, dim=1), torch.cat(F_dots, dim=1)