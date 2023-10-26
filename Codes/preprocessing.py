import numpy as np
from data import TorchGraphData
import torch
import nxmetis
from typing import List, Union
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
# from torch import Tensor
# from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler

##############################################################################
def get_graph_partition(
    data : TorchGraphData,
    partition : np.array,
    recursive : bool,
    list_of_node_features = ['node_attr', 'pressure', 'flowrate', 'flowrate_bc'],
    list_of_edge_features = ['edge_attr']
) -> TorchGraphData:
    '''
    Return sub-graph data for given list of node index in a parition
    '''
    edge_index = data.edge_index.numpy()

    # Mark all edges containing nodes in partition
    edge_mark = np.isin(edge_index, partition)
    if recursive:
        edge_mark = np.logical_or(edge_mark[0], edge_mark[1])
    else:
        edge_mark = np.logical_and(edge_mark[0], edge_mark[1])
    # Get global id of all edges containing nodes in partition
    partition_edge_id = np.argwhere(edge_mark == True).squeeze(1)
    # Get edge_index of partition (with global node id)
    partition_edge_index = edge_index[:, partition_edge_id]
    # Get global id of all nodes in parition (for recursive only)
    partition_node_id = np.unique(np.concatenate(list(partition_edge_index) + [partition]))

    #### Convert global node id to partition node id
    # Lambda
    index = lambda n : list(partition_node_id).index(n)
    v_index = np.vectorize(index)
    # Convert global node id to partition node id in partition_edge_index
    if partition_edge_index.shape[1] > 0:
        partition_edge_index = torch.tensor(v_index(partition_edge_index))
    
    #### Get partition of all features
    partition_data = TorchGraphData()
    for key in data._store:
        if key in list_of_node_features:
            setattr(partition_data, key, data._store[key][partition_node_id])
        elif key in list_of_edge_features:
            setattr(partition_data, key, data._store[key][partition_edge_id])
        elif key == 'edge_index':
            setattr(partition_data, key, partition_edge_index)
        else:
            setattr(partition_data, key, data._store[key])
    return partition_data



##############################################################################
def get_time_partition(
    data : TorchGraphData,
    time_partition : np.array,
    list_of_time_features = ['pressure', 'flowrate', 'velocity', 'flowrate_bc', 'pressure_dot', 'flowrate_dot', 'time']
) -> TorchGraphData:
    partition_data = TorchGraphData()
    for key in data._store:
        if key in list_of_time_features:
            setattr(partition_data, key, data._store[key][:, time_partition])
        else:
            setattr(partition_data, key, data._store[key])
    return partition_data



##############################################################################
def get_batch_graphs(
    data : TorchGraphData,
    batch_size : int = None,
    batch_n_times : int = None,
    recursive : bool = False,
    step : int = 1
) -> List[TorchGraphData]:
    
    # Graph partitioning
    temp_list_of_partitions = []
    if batch_size is not None:
        if batch_size <= 100:
            list_of_partitions = BFS_partition(
                edge_index=data.edge_index.numpy(),
                partition_size=batch_size,
                threshold=0.3
            )
        else:
            (_, list_of_partitions) = nxmetis.partition(
                G=data.graph,
                nparts=int(data.number_of_nodes / batch_size)
            )
        for partition in list_of_partitions:
            temp_list_of_partitions.append(get_graph_partition(data, partition, recursive))
    else:
        temp_list_of_partitions.append(data)
    
    # Graph time partitioning/slicing
    list_of_partitions = []
    list_of_time_partitions = []
    i = 0
    if batch_n_times is None:
        batch_n_times = data.number_of_timesteps
    while i < data.number_of_timesteps - 1:
        i_end = i + batch_n_times + recursive
        list_of_time_partitions.append(
            np.arange(
                start=i,
                stop=min(i_end, data.number_of_timesteps),
                step=step,
                dtype=int
            )
        )
        i = i_end - recursive

    for data_partition in temp_list_of_partitions:
        for time_partition in list_of_time_partitions:
            list_of_partitions.append(get_time_partition(data_partition, time_partition))
    
    return list_of_partitions



###########################################################################
def BFS_partition(edge_index, partition_size=None, n_partitions=None, threshold=0.3) -> List[List[int]]:
    '''
    Perform breath first search to partition a graph(given by edge index).
    '''
    def BFS(edge_index, root, visited, part_size=100):
        queue = [root]
        visited = []
        part = []
        while queue:
            current = queue.pop(0)
            while current in visited:
                current = queue.pop(0)
            visited.append(current)
            part.append(current)
            # find child nodes
            child_edges = np.where(edge_index[0] == current)[0]
            child_nodes = list(edge_index[1][child_edges])
            # add child nodes to queue
            queue += child_nodes
            # break
            if len(part) >= part_size:
                break
        return part, queue, visited
    
    if partition_size is None:
        partition_size = int((edge_index.shape[1] + 1)/n_partitions)
    root_queue = [0]
    visited = []
    partitions = []
    root_parrent = [None]
    
    while root_queue:
        root = root_queue.pop(0)
        parrent = root_parrent.pop(0)
        partition, queue, visited = BFS(edge_index, root, visited, partition_size)
        root_queue += queue
        root_parrent += [len(partitions)] * len(queue)
        if len(partition) >= threshold*partition_size:
            partitions.append(partition)
        else:
            partitions[parrent] += partition
    return partitions


##############################################################################
def dataset_split_to_loader(
    dataset : Dataset,
    subset_ids = {
        'train': [],
        'test': [],
        'validation': []
    },
    n_datas_per_batch = 1

):
    # Get batching id
    if 'batched' in dataset.sub:
        batching_id = dataset.batching_id.numpy()
        for subset in subset_ids:
            subset_ids[subset] = list(np.where(np.isin(batching_id, subset_ids[subset] ) == True)[0])
        # train_ids = list(np.where(np.isin(batching_id, train_ids) == True)[0])
        # eval_ids = list(np.where(np.isin(batching_id, eval_ids) == True)[0])
    subsets = []
    for subset in subset_ids:
        subsets.append(DataLoader([dataset[i] for i in subset_ids[subset]], batch_size=n_datas_per_batch))
    return tuple(subsets)