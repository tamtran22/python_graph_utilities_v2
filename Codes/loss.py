import torch
import torch.nn as nn
from torch import Tensor
from data import TorchGraphData
# from dataset import edge_to_node, node_to_edge
from utils import NodeToEdgeLayer, EdgeToNodeLayer
import numpy as np


def kinetic_energy_coefficient(generation: int, reynold_number: float,
    is_inspiration: bool) -> float:
    if generation < 0:
        return 0
    if is_inspiration:
        if generation <= 8:
            return 0.8519+6.42*(10**-2)*generation-5.456*(10**-5)*(generation*reynold_number)
        else:
            return 1.33
    else: # expiration
        return 0.8277+3.696*(10**-2)*generation-2.671*(10**-5)*(generation*reynold_number)


def mse_pinn_loss(data: TorchGraphData) -> Tensor:
    node_to_edge_1 = NodeToEdgeLayer(message=lambda u,v: u-v)
    node_to_edge_2 = NodeToEdgeLayer(message=lambda u,v: u)

    pressure_uv = node_to_edge_1(data.pressure, data.edge_index)
    flowrate_uv = node_to_edge_2(data.flowrate, data.edge_index)

    length = data.edge_attr[:,0]/1000
    diameter = data.edge_attr[:,1]/1000
    generation = data.edge_attr[:,2]
    rho = data.rho
    nu = data.vis
    mu = rho*nu
    pi = np.pi
    gamma = 0.327
    alpha = 0.5

    ## Pressure losss
    p_loss = pressure_uv


    ## Kinetic energy loss
    # calculate Reynolds for all edge
    Re = (4*rho/(pi*mu))*torch.einsum('ij,i->ij',torch.abs(flowrate_uv),1./diameter)
    print(Re)
    # calculate kinetic energy coefficient
    K_inspiration_1 = 0.8519 \
                +6.42*(10**-2)*torch.unsqueeze(generation, dim=-1) \
                -5.456*(10**-5)*torch.einsum('ij,i->ij',Re, generation)
    K_inspiration_2 = torch.full(size=K_inspiration_1.size(), fill_value=1.33)
    K_inspiration = torch.einsum('ij,i->ij',K_inspiration_1, generation<=8)+\
                    torch.einsum('ij,i->ij',K_inspiration_2, generation>8)
    
    K_expiration = 0.8277 \
                +3.696*(10**-2)*torch.unsqueeze(generation, dim=-1) \
                -2.671*(10**-5)*torch.einsum('ij,i->ij',Re, generation)
    
    time = (4.0/200)*torch.arange(start=0, end=201, step=1)

    K = torch.einsum('ij,j->ij',K_inspiration, time<data.total_time[0].item()/2)+\
        torch.einsum('ij,j->ij',K_expiration, time>=data.total_time[0].item()/2)
    # calculate kinetic term
    Kin = (16*rho/(pi**2))*torch.einsum('ij,i->ij', K, 1./torch.square(torch.square(diameter)))
    # calculate loss
    k_loss = torch.einsum('ij,ij->ij', Kin, torch.square(flowrate_uv))


    ## Viscous loss
    Z = gamma*torch.pow(torch.einsum('ij,i->ij',Re,diameter/length),alpha)
    Vis_1 = (128*mu/pi)*(length/torch.square(torch.square(diameter)))
    Vis_2 = (2*gamma/torch.sqrt(pi*nu))*torch.pow(torch.einsum('ij,i->ij',torch.abs(flowrate_uv),1./length),0.5)
    Vis_2 = torch.einsum('i,ij->ij',Vis_1,Vis_2)
    Vis = torch.einsum('i,ij->ij',Vis_1,Z<=1.)+torch.einsum('ij,ij->ij',Vis_2,Z>1)

    v_loss = torch.einsum('ij,ij->ij', Vis, torch.abs(flowrate_uv))

    ## Unsteady loss
    delta_t = 4.0 / 200
    Uns = (4*rho/pi) * (length / torch.square(diameter))
    delta_Q = flowrate_uv[:,:-1]-flowrate_uv[:,1:]
    delta_Q = torch.cat([torch.zeros((delta_Q.size(0),1)),delta_Q],dim=1)


    u_loss = torch.einsum('i,ij->ij',Uns,torch.abs(delta_Q)/delta_t)

    loss = p_loss+k_loss+v_loss+u_loss
    return p_loss, k_loss, v_loss, u_loss, loss


# class OneDAirwayLoss(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
    
#     def forward(self, input_data: TorchGraphData):
#         # k_loss = kinematic_loss(input_data)[:,1:]
#         # v_loss = viscous_loss(input_data)[:,1:]
#         # u_loss = unsteady_loss(input_data)
#         # p_loss = pressure_loss(input_data)[:,1:]
#         # # print(k_loss.size(), v_loss.size(), u_loss.size(), p_loss.size())
#         # loss = k_loss + v_loss + u_loss + p_loss
#         # return loss