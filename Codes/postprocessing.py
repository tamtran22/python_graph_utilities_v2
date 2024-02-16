import numpy as np
import os
import pickle

def print_tecplot_file(
        nodes={
            'x': np.array([1.,2,3,4]),
            'y': np.array([6.,7,8,9]),
            'z': np.array([1,2,4,9.2])
        },
        elements=np.array([[1,2],[2,3], [2,4]]), 
        ZONE_T='output.dat', 
        F='FEPoint', 
        ET='LINESEG',
        file_name='output.dat'
    ):
    # check input
    try:
        n_field = len(list(nodes)) # number of fields
        for field in nodes:
            if len(np.shape(nodes[field])) != 1:
                raise KeyError
        n_node = np.shape(nodes[list(nodes.keys())[0]])[0] # number of nodes
        for i in range(1, n_field):
            if n_node != np.shape(nodes[list(nodes.keys())[i]])[0]:
                raise KeyError
        if len(np.shape(elements)) != 2:
            raise KeyError
        n_element = np.shape(elements)[0] # number of elements
        n_node_per_element = np.shape(elements)[1]
    except:
        raise KeyError

    # start
    f = open(file_name, 'w+')
    # header
    f.write('VARIABLES=')
    for field in nodes:
        f.write(f'"{field}" ')
    f.write('\n')
    f.write(f'ZONE T= "{ZONE_T}"\n')
    f.write(f'N={n_node}, E={n_element}, F={F}, ET={ET}\n')
    # data
    for i in range(n_node):
        for field in nodes:
            f.write(f' {nodes[field][i]}')
        f.write('\n')
    
    for i in range(n_element):
        for j in range(n_node_per_element):
            f.write(f' {elements[i][j]}')
        f.write('\n')
    # finish
    f.close()

def print_1D(data, dir='./test'):
    os.system(f'mkdir {dir}')
    n_time = data.pressure.size(1)
    for i in range(n_time):
        print_tecplot_file(
            nodes={
                'x': data.node_attr[:,0].numpy(),
                'y': data.node_attr[:,1].numpy(),
                'z': data.node_attr[:,2].numpy(),
                'pressure': data.pressure[:,i].numpy(),
                'flowrate': data.flowrate[:,i].numpy()
            },
            elements=data.edge_index.numpy().transpose() + 1,
            ZONE_T=f'plt_nd_{str(i).zfill(6)}.dat',
            file_name=dir+f'/plt_nd_{str(i).zfill(6)}.dat'
        )

def reverse_scaler(data, scalers):
    for field in data.__dict:
        if f'{field}_scaler' in scalers:
            pass