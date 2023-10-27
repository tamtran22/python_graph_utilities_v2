import torch
from data import TorchGraphData
from networks_lstm import MessageNet
import torch.nn.functional as F

def train(model, data : TorchGraphData, args):

    ## Number of timesteps of the current data (might be batched data)
    model.n_timesteps = data.pressure.size(1)
    timestep = args.total_time / model.n_timesteps
    
    ## Connectivity/edge_index: Tensor(2, n_edges)
    edge_index = torch.cat([
        data.edge_index, 
        torch.flip(data.edge_index, dims=[0]
    )], dim=1).to(args.device)
    
    ## Mesh features: Tuple(NodeTensor, EdgeTensor)
    node_attr = data.node_attr.to(args.device).float()           
    edge_attr = torch.cat([data.edge_attr.to(args.device).float()]*2,dim=0)
    mesh_features = (node_attr, edge_attr)
    
    ## Fields tensor(pressure, flowrate): Tensor(n_nodes, n_times, n_fields)
    F_initial = torch.cat([
        data.pressure[:,0].unsqueeze(-1), 
        data.flowrate[:,0].unsqueeze(-1)
    ], dim=-1).to(args.device).float() # concat pressure and flowrate

    ## Boundary value tensor (flowrate at entrance): Tensor(n_nodes, n_times)
    F_bc = torch.zeros((data.number_of_nodes, model.n_timesteps))
    F_bc[0,:] = data.flowrate[0,:]
    F_bc = F_bc.to(args.device).float()

    ## Predict output
    Fs, _ = model(
        F_initial=F_initial, 
        mesh_features=mesh_features, 
        edge_index=edge_index, 
        F_boundary=F_bc, 
        timestep=timestep
    )

    ## Ground truth Fields tensor(pressure, flowrate): Tensor(n_nodes, 1:n_times, n_fields)
    Fs_hat = torch.cat([
        data.pressure.unsqueeze(-1), 
        data.flowrate.unsqueeze(-1)
    ], dim=-1).to(args.device).float() # concat pressure and flowrate
    Fs_hat = Fs_hat[:,1:,:]
    
    ## PINN
    message_net = MessageNet().to(args.device)
    # P_pred = torch.cat([F_initial[:,0].unsqueeze(1), Fs[:,:,0]], dim=1)
    # Q_pred = torch.cat([F_initial[:,1].unsqueeze(1), Fs[:,:,1]], dim=1)
    # P_pred = message_net(P_pred, edge_index)
    # Q_pred = message_net(Q_pred, edge_index)
    
    P_true = torch.cat([F_initial[:,0].unsqueeze(1), Fs_hat[:,:,0]], dim=1)
    Q_true = torch.cat([F_initial[:,1].unsqueeze(1), Fs_hat[:,:,1]], dim=1)
    # with torch.no_grad():
    #     P_true = message_net(P_true, edge_index)
    #     Q_true = message_net(Q_true, edge_index)
    L = edge_attr[:,0]
    D = edge_attr[:,1]
    # # pinn_pred = kinematic_loss(P_pred, Q_pred, L, D)
    # kin_true = kinematic_loss(P = P_true,Q= Q_true,L= L,D= D)
    # vis_true = viscous_loss(P=P_true,Q= Q_true,L= L,D= D)
    # uns_true = unsteady_loss(P=P_true,Q= Q_true,L= L,D= D, dt=timestep)
    # # print(args.criterion(pinn_true, pinn_pred))
    # print(kin_true.size(), vis_true.size(), uns_true.size(), P_true[:,1:].size())
    # print(args.criterion(kin_true, torch.zeros_like(kin_true)),
    #       args.criterion(vis_true, torch.zeros_like(vis_true)),
    #       args.criterion(uns_true, torch.zeros_like(uns_true)),
    #       args.criterion(P_true[:,1:], torch.zeros_like(P_true[:,1:])))

    with torch.no_grad():
        loss_true = unsteady(
            P = message_net(P_true, edge_index, message = lambda a,b: a-b),
            Q = message_net(Q_true, edge_index, message = lambda a,b: a-b),
            L = L,
            D = D,
            dt=4.8/200
        )
        print(args.criterion(loss_true, torch.zeros_like(loss_true)))

    ## Loss function
    loss = args.criterion(Fs_hat, Fs)
    loss.backward()
    args.optimizer.step()

    return loss.item()




def eval(model, data, args):
    
    ## Number of timesteps of the current data (might be batched data)
    model.n_timesteps = data.pressure.size(1)
    timestep = args.total_time / model.n_timesteps
    
    ## Connectivity/edge_index: Tensor(2, n_edges)
    edge_index = torch.cat([
        data.edge_index, 
        torch.flip(data.edge_index, dims=[0]
    )], dim=1).to(args.device)
    
    ## Mesh features: Tuple(NodeTensor, EdgeTensor)
    mesh_features = (
        data.node_attr.to(args.device).float(),            
        torch.cat([data.edge_attr.to(args.device).float()]*2,dim=0)
    )
    
    ## Fields tensor(pressure, flowrate): Tensor(n_nodes, n_times, n_fields)
    F_initial = torch.cat([
        data.pressure[:,0].unsqueeze(1), 
        data.flowrate[:,0].unsqueeze(1)
    ], dim=-1).to(args.device).float() # concat pressure and flowrate

    ## Boundary value tensor (flowrate at entrance): Tensor(n_nodes, n_times)
    F_bc = torch.zeros((data.number_of_nodes, model.n_timesteps))
    F_bc[0,:] = data.flowrate[0,:]
    F_bc = F_bc.to(args.device).float()

    ## Predict output
    with torch.no_grad():
        Fs, F_dots = model(
            F_initial=F_initial, 
            mesh_features=mesh_features, 
            edge_index=edge_index, 
            F_boundary=F_bc, 
            timestep=timestep
        )

        ## Ground truth Fields tensor(pressure, flowrate): Tensor(n_nodes, 1:n_times, n_fields)
        Fs_hat = torch.cat([
            data.pressure.unsqueeze(-1), 
            data.flowrate.unsqueeze(-1)
        ], dim=-1).to(args.device).float() # concat pressure and flowrate
        Fs_hat = Fs_hat[:,1:,:]
        
        ## Ground truth Fields time derivative tensor

        ## Loss function
        loss = args.criterion(Fs_hat, Fs)

    return loss.item()




def kinematic(P, Q, L, D, rho: float = 1.12, pi: float = 3.1415926, mu: float = 1.64E-5, K: float = 1.
):
    Kin = (16*K*rho) / ((pi**2) * D * D * D * D)
    loss = torch.transpose(torch.mul(torch.transpose(Q[:,:-1]*Q[:,:-1], 0, 1), Kin), 0, 1)

    return loss

def unsteady(P, Q, L, D, rho: float = 1.12, pi: float = 3.1415926, mu: float = 1.64E-5, K: float = 1., dt:float=0.1
):
    Uns = (4*rho*L) / (pi * D * D)
    loss = (1./dt) * torch.transpose(torch.mul(torch.transpose(Q[:,1:]-Q[:,:-1],0,1),Uns),0,1)
    return loss

def viscous(P, Q, L, D, rho: float = 1.12, pi: float = 3.1415926, mu: float = 1.64E-5, K: float = 1.
):
    Vis = (128*mu*L) / (pi * D * D * D * D)
    loss = torch.transpose(torch.mul(torch.transpose(Q[:,1:], 0, 1), Vis), 0, 1)
    return loss

def boundary_loss(
):
    pass