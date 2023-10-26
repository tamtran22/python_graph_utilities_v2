import matplotlib.pyplot as plt
from data import TorchGraphData
import numpy as np
from dataset import edge_to_node

def plot_path(data: TorchGraphData, var_name: str, var_column: int, ending_node_id: int, plot: bool = False):
    edge_index = data.edge_index.numpy()
    field = data._store[var_name][:, var_column]
    if var_name == 'edge_attr':
        field = edge_to_node(field, edge_index)

    current_node_id = ending_node_id
    path = []
    while True:
        path.append(current_node_id)
        current_edge_ids = np.where(edge_index[1] == current_node_id)[0]
        if len(current_edge_ids) < 1:
            break
        current_edge_id = current_edge_ids[0]
        current_node_id = edge_index[0][current_edge_id]

    path = list(reversed(path))
    field_path = np.array([field[i] for i in path])
    plt.plot(field_path)
    return field_path, path