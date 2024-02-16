import torch
from data import TorchGraphData

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
    mesh_features = (
        data.node_attr.to(args.device).float(),            
        torch.cat([data.edge_attr.to(args.device).float()]*2,dim=0)
    )
    
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
