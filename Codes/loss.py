import torch
import torch.nn as nn
from torch import Tensor
from data import TorchGraphData
# from dataset import edge_to_node, node_to_edge
from utils import NodeToEdgeLayer, EdgeToNodeLayer

def kinematic_loss(data: TorchGraphData) -> Tensor:
    edge_to_node = EdgeToNodeLayer()
    diam = edge_to_node(data.edge_attr[:,1], data.edge_index)
    diam[0] = diam[1] # set root value
    Q = data.flowrate # node wise features
    rho = data.rho
    pi = 3.1415926
    K=1. # temporary...

    Kin = (16*K*rho) / ((pi**2)*torch.square(torch.square(diam)))
    loss = torch.einsum('i, ij->ij', Kin, torch.square(Q))
    return torch.mean(loss)

def viscous_loss(data: TorchGraphData) -> Tensor:
    node_to_edge = NodeToEdgeLayer(message=lambda a,b: a-b)
    Q = node_to_edge(data.flowrate, data.edge_index) # edge wise Q
    mu = data.vis
    pi = 3.1415926
    length = data.edge_attr[:,0]
    diam = data.edge_attr[:,1]

    Vis = (128*mu*length) / (pi * torch.square(torch.square(diam)))
    loss = torch.einsum('i, ij->ij', Vis, Q)
    return torch.mean(loss)

def unsteady_loss(data: TorchGraphData) -> Tensor:
    node_to_edge = NodeToEdgeLayer(message=lambda a,b: a-b)
    Q = node_to_edge(data.flowrate, data.edge_index)
    mu = data.vis
    rho = data.rho
    pi = 3.1415926
    timestep = data.total_time / (data.flowrate.size(1)-1)
    length = data.edge_attr[:,0]
    diam = data.edge_attr[:,1]

    Uns = (4*rho*length) / (pi * torch.square(diam))
    delta_Q = Q[:,1:] - Q[:,0:-1]
    loss = torch.einsum('i, ij->ij',Uns / timestep,delta_Q)
    return torch.mean(loss)

def pressure_loss(data: TorchGraphData) -> Tensor:
    node_to_edge = NodeToEdgeLayer(message=lambda a,b: a-b)
    loss = node_to_edge(data.pressure, data.edge_index)
    return torch.mean(loss)



class OneDAirwayLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, input_data: TorchGraphData):
        loss = kinematic_loss(input_data)
        loss += viscous_loss(input_data)
        loss += unsteady_loss(input_data)
        loss += pressure_loss(input_data)
        return torch.mean(loss)