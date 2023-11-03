import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from typing import Optional, Callable, Union, List, Tuple
from preprocessing import *
from data import TorchGraphData
from sklearn.preprocessing import PowerTransformer
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
        

# Graph data builder (set)
########################################################################
class OneDDatasetBuilder(Dataset):
    r'''
    Graph data class expanded from torch_geometric.data.Dataset()
    Build and store multiple graph datas
    '''
    def __init__(self,
        raw_dir: Optional[str] = None, # Path to raw data files
        root_dir: Optional[str] = None, # Path to store processed data files
        sub_dir: Optional[str] = 'processed',
        data_names: Optional[Union[str, List[str], Tuple]] = 'all',
        time_names: Optional[Union[str, List[str], Tuple]] = 'all',
        has_time_features: bool = 'True',
        data_type = torch.float32
    ):
        transform = None
        pre_transform = None
        pre_filter = None
        self.raw = raw_dir
        self.sub = sub_dir
        self.has_time_features = has_time_features
        self.data_type = data_type

        self._set_data_names(data_names, raw_dir)
        self._set_time_names(time_names)
        super().__init__(root_dir, transform, pre_transform, pre_filter)

    def _set_data_names(self, data_names, data_dir):
        if (data_names == 'all') or (data_names is None):
            self.data_names = os.listdir(data_dir)
        else:
            self.data_names = data_names
    
    @property
    def processed_file_names(self):
        return [f'{self.root}/{self.sub}/{data}.pt' for data in self.data_names]

    def _set_time_names(self, time_names):
        self.time_names = time_names

    # @property
    def len(self):
        return len(self.data_names)
    
    def __getitem__(self, index):
        return torch.load(self.processed_file_names[index])
    
    def get(self, index):
        return torch.load(self.processed_file_names[index])
    
    def process(self):
        # Declare data file name
        CFD_1D_dir = 'CFD_1D'
        file_name_input = lambda subject : f'{self.raw}/{subject}'+\
            f'/{CFD_1D_dir}/Output_{subject}_Amount_St_whole.dat'
        file_name_output = lambda subject, time : f'{self.raw}/{subject}'+\
            f'/{CFD_1D_dir}/data_plt_nd/plt_nd_000{time}.dat'

        # Read data
        for subject in self.data_names:
            print(f'Process subject number {self.data_names.index(subject)}, subject name : {subject}.')

            data_dict_input = read_1D_input(file_name_input(subject))

            _file_name_outputs = [file_name_output(subject, time) for time in self.time_names]
            data_dict_output = read_1D_output(_file_name_outputs)

            # Value of time steps
            # if self.has_time_features:
            #     _total_time = 4.8
            #     _n_time = len(self.time_names)
            #     _time = (_total_time/(_n_time - 1)) * np.arange(start=0, stop=_n_time, step=1)
            #     time = torch.tensor(_time).type(self.data_type).unsqueeze(0)
            # else: 
            #     time = None

            # List of variables
            edge_index = torch.tensor(data_dict_input['edge_index']).type(torch.LongTensor)
            edge_attr = torch.tensor(data_dict_input['edge_attr']).type(self.data_type)
            node_attr = torch.tensor(data_dict_input['node_attr']).type(self.data_type)
            pressure = torch.tensor(data_dict_output['pressure']).type(self.data_type)
            flowrate = torch.tensor(data_dict_output['flowrate']).type(self.data_type)
            is_terminal = torch.tensor(data_dict_input['is_terminal']).type(torch.LongTensor)
            

            # Convert node to edge
            # edge_attr = node_to_edge(edge_attr, edge_index)

            data = TorchGraphData(edge_index=edge_index, edge_attr=edge_attr,
                node_attr=node_attr, pressure=pressure, flowrate=flowrate, 
                is_terminal=is_terminal)
            torch.save(data, self.processed_file_names[self.data_names.index(subject)])






class OneDDatasetLoader(Dataset):
    r'''
    Graph data class expanded from torch_geometric.data.Dataset()
    Load multiple graph datas
    '''
    def __init__(self,
        raw_dir: Optional[str] = None, # Path to raw data files
        root_dir: Optional[str] = None, # Path to store processed data files
        sub_dir: Optional[str] = 'processed',
        data_names: Optional[Union[str, List[str], Tuple]] = 'all',
        time_names: Optional[Union[str, List[str], Tuple]] = 'all',
        data_type = torch.float32
    ):
        transform = None
        pre_transform = None
        pre_filter = None
        self.raw = raw_dir
        self.sub = sub_dir
        self.data_type = data_type

        self._set_data_names(data_names, f'{root_dir}/{sub_dir}')
        self._set_time_names(time_names)
        super().__init__(root_dir, transform, pre_transform, pre_filter)

    def _set_data_names(self, data_names, data_dir):
        if (data_names == 'all') or (data_names is None):
            data_names = os.listdir(data_dir)
            f_filter = lambda s : not s in [
                'pre_filter.pt', 'pre_transform.pt', 'batched_id.pt', 'batched_info.pt', \
                'edge_attr_scaler.pkl', 'node_attr_scaler.pkl', 'pressure_scaler.pkl', 'flowrate_scaler.pkl']
            data_names = list(filter(f_filter, data_names))
            data_names = [data.replace('.pt','',data.count('.pt')) for data in data_names]
            self.data_names = data_names
        else:
            self.data_names = data_names
    
    @property
    def processed_file_names(self):
        return [f'{self.root}/{self.sub}/{data}.pt' for data in self.data_names]

    def _set_time_names(self, time_names):
        self.time_names = time_names
    
    # @property
    def len(self):
        return len(self.data_names)
    
    def __getitem__(self, index):
        return torch.load(self.processed_file_names[index])
    
    def get(self, index):
        return torch.load(self.processed_file_names[index])

    def process(self):
        pass
    
    def batching(self, batch_size : int, batch_n_times : int, recursive : bool, sub_dir='batched', step=1):
        ''' Perform batching and return batched dataset.
        batch_size : approximate size of sub-graph datas.
        batch_n_times : number of timesteps in sub-graph datas.
        recursive : indicator to partition recursive sub-graphs.
        sub_dir : sub folder to store batched dataset.
        '''
        self._clean_sub_dir(sub_dir)
        os.system(f'mkdir {self.root}{sub_dir}')
        batched_dataset = []
        batched_dataset_id = []
        for i in range(self.len()):
            batched_data = get_batch_graphs(
                data=self.__getitem__(i),
                batch_size=batch_size,
                batch_n_times=batch_n_times,
                recursive=recursive,
                step=step
            )
            batched_dataset += batched_data
            batched_dataset_id += [i]*len(batched_data)
            
        for i in range(len(batched_dataset)):
            torch.save(batched_dataset[i], f'{self.root}/{sub_dir}/batched_data_{i}.pt')
        
        torch.save(torch.tensor(batched_dataset_id), f'{self.root}/{sub_dir}/batched_id.pt')
        torch.save({'batch_size' : batch_size, 'batch_n_times':batch_n_times}, f'{self.root}/{sub_dir}/batched_info.pt')
        return OneDDatasetLoader(root_dir=self.root, sub_dir=sub_dir)
    
    def _clean_sub_dir(self, sub_dir='batched'):
        ''' Clear the sub folder to store new processed data.
        '''
        if sub_dir == '' or sub_dir == '/':
            print('Unable to clear root folder!')
        else:
            os.system(f'rm -rf {self.root}/{sub_dir}')

    @property
    def batching_id(self):
        ''' Map batched data index which is stored in batched dataset to original 
        data index which is stored in raw processed dataset.
        In case of dataset is batched (_sub_dir==/batched), return an array which index is
        the index of batched data (in data_names) and value is the index of parrent data.
        In case of dataset is not batched, return zero tensor.
        '''
        try:
            return torch.load(f'{self.root}/{self.sub}/batched_id.pt')
        except:
            return torch.tensor(0)

    def get_scaler(self, scaler_type=None, var_name=None, axis=None):
        if (var_name is None) or (scaler_type is None):
            return None
        var = []
        for i in range(self.len()):
            var.append(self.__getitem__(i)._store[var_name])
        var = torch.cat(var, dim=0)
        if axis is None:
            var = var.flatten().unsqueeze(1)
        if scaler_type=='minmax_scaler':
            scaler = MinMaxScaler()
            scaler.fit(var)
            return scaler
        if scaler_type=='standard_scaler':
            scaler = StandardScaler()
            scaler.fit(var)
            return scaler
        if scaler_type=='power_transformer':
            scaler = PowerTransformer()
            scaler.fit(var)
            return scaler
        if scaler_type=='robust_scaler':
            scaler = RobustScaler()
            scaler.fit(var)
            return scaler
        if scaler_type=='quantile_transformer':
            scaler = QuantileTransformer(output_distribution='normal')
            scaler.fit(var)
            return scaler

    def normalizing(self, 
        sub_dir='normalized',
        scalers = {
            'node_attr' : ['minmax_scaler', 0],
            'edge_attr' : ['quantile_transformer', 0],
            'pressure' : ['quantile_transformer', None],
            'flowrate' : ['quantile_transformer', None]
        }
    ):
        ''' Perform normalizing and return normalized dataset.
        sub_dir : sub folder to store normalized dataset.
        sc
        '''
        self._clean_sub_dir(sub_dir)
        os.system(f'mkdir {self.root}/{sub_dir}')
        if not os.path.isdir(f'{self.root}/scalers'):
            os.system(f'mkdir {self.root}/scalers')
        if self.len() <= 0:
            return self
        
        ### Create scalers
        for var_name in scalers:
            _scaler = self.get_scaler(scaler_type=scalers[var_name][0], var_name=var_name, axis=scalers[var_name][1])
            _file_picle = open(f'{self.root}/scalers/{var_name}_scaler.pkl','wb')
            pickle.dump(_scaler, _file_picle)
            _file_picle.close()
        ### Normalized data
        for i in range(self.len()):
            data = self.__getitem__(i)
            normalized_data = TorchGraphData()
            for var_name in data._store:
                if (var_name in scalers):
                    var_data = data._store[var_name]
                    if scalers[var_name][1] is not None:
                        var_data = self.scaler(var_name).transform(var_data)
                        var_data = torch.tensor(var_data)
                    else:
                        _size = var_data.size()
                        var_data = var_data.flatten().unsqueeze(1)
                        var_data = self.scaler(var_name).transform(var_data)
                        var_data = torch.tensor(var_data)
                        var_data = torch.reshape(var_data, _size)
                else:
                    # Default, not scaled
                    var_data = data._store[var_name]
                setattr(normalized_data, var_name, var_data)
            
            torch.save(normalized_data, f'{self.root}/{sub_dir}/{self.data_names[i]}.pt')
        return OneDDatasetLoader(root_dir=self.root, sub_dir=sub_dir)

    def scaler(self, var_name = None):
        if var_name is None:
            return None
        if not os.path.isdir(f'{self.root}/scalers'):
            return None
        _file_picle = open(f'{self.root}/scalers/{var_name}_scaler.pkl','rb')
        _scaler = pickle.load(_file_picle)
        _file_picle.close()
        return _scaler
        






# Read 1D data into graph
########################################################################

def read_1D_input(
        file_name : str,
        # var_dict = {
        #     'node_attr' : ['x_end', 'y_end', 'z_end'], 
        #     'edge_index' : ['PareID', 'ID'], 
        #     'edge_attr' : ['Length', 'Diameter', 'Gene', 'Lobe', 'Flag', 'Vol0', 'Vol1', 'Vol1-0']
        # },
        var_dict = {
            'node_attr' : ['x_end', 'y_end', 'z_end'], 
            'edge_index' : ['PareID', 'ID'], 
            'edge_attr' : ['Length', 'Diameter', 'Gene', 'Lobe', 'Vol0', 'Vol1', 'Vol1-0'],
            'is_terminal' : ['Flag']
        },
        # var_dict = {
        #     'node_attr' : ['x_end', 'y_end', 'z_end', 'Length', 'Diameter', 'Gene', 'Lobe', 'Flag', 'Vol0', 'Vol1'], 
        #     'edge_index' : ['PareID', 'ID']
        # }
    ):
    r"""Read Output_subject_Amount_St_whole.dat
    Data stored in edge-wise format
    Data format
    ID PareID Length Diameter ... Vol1-0 Vol0 Vol1
    -  -      -      -        ... -      -    -
    -  -      -      -        ... -      -    -
    (---------information of ith branch----------)
    -  -      -      -        ... -      -    -
    """
    # print(var_dict)
    def _float(str):
        # _dict = {'C':0, 'P':1, 'E':2, 'G':3, 'T':4}
        _dict = {'C':0, 'P':0, 'E':0, 'G':0, 'T':1}
        try:
            return float(str)
        except:
            return _dict[str]
    _vectorized_float = np.vectorize(_float)

    file = open(file_name, 'r')
    # Read header
    header = file.readline()
    # Read data
    data = file.read()
    # Done reading file
    file.close()

    # Process header
    vars = header.replace('\n',' ')
    vars = vars.split(' ')
    vars = list(filter(None, vars))

    n_var = len(vars)

    # Process data
    data = data.replace('\n',' ')
    data = data.split(' ')
    data = list(filter(None, data))

    data = np.array(data).reshape((-1, n_var)).transpose()
    data_dict = {}
    for i in range(len(vars)):
        data_dict[vars[i]] = _vectorized_float(data[i])
    
    # Rearange data
    data_dict['x_end'] = np.insert(data_dict['x_end'], 0, data_dict['x_start'][0])
    data_dict['y_end'] = np.insert(data_dict['y_end'], 0, data_dict['y_start'][0])
    data_dict['z_end'] = np.insert(data_dict['z_end'], 0, data_dict['z_start'][0])

    # Scaling data - cubic root of volume
    if data_dict['Vol0'] is not None:
        data_dict['Vol0'] = np.cbrt(data_dict['Vol0']) 
    if data_dict['Vol1'] is not None:
        data_dict['Vol1'] = np.cbrt(data_dict['Vol1']) 
    if data_dict['Vol1-0'] is not None:
        data_dict['Vol1-0'] = np.cbrt(data_dict['Vol1-0']) 

    out_dict = {}
    for var in var_dict:
        out_dict[var] = []
        for data_var in var_dict[var]:
            out_dict[var].append(data_dict[data_var])
        if len(out_dict[var]) == 1:
            out_dict[var] = out_dict[var][0]
    out_dict['edge_index'] = np.array(out_dict['edge_index'], dtype=np.int32)
    out_dict['edge_attr'] = np.array(out_dict['edge_attr'], dtype=np.float32).transpose()
    out_dict['node_attr'] = edge_to_node(np.array(out_dict['node_attr'], dtype=np.float32).transpose(),
                                        out_dict['edge_index'])
    out_dict['is_terminal'] = edge_to_node(np.array(out_dict['is_terminal'], dtype=int).transpose(),
                                        out_dict['edge_index'])
    return out_dict

def read_1D_output(
        file_names,
        var_dict = {
            'pressure' : 'p',
            'flowrate' : 'flowrate'
        }
    ):
    r"""Read data_plt_nd/plt_nd_000time.dat (all time_id)
    Data stored in node wise format
    Data format
    VARIABLES="x" "y" "z" "p" ... "flowrate"  "resist" "area"                                    
     ZONE T= "plt_nd_000time.dat                                 "
     N=       xxxxx , E=       xxxxx ,F=FEPoint,ET=LINESEG
    -  -      -      -        ... -      -    -
    -  -      -      -        ... -      -    -
    (---------information of ith node----------)
    -  -      -      -        ... -      -    -
    -  -
    -  -
    (---------connectivity of jth branch-------)
    -  -
    """
    # Read variable list and n_node, n_edge
    file = open(file_names[0], 'r')
    line = file.readline()
    line = line.replace('VARIABLES',' ')
    line = line.replace('=',' ')
    line = line.replace('\n',' ')
    line = line.replace('"',' ')
    vars = list(filter(None, line.split(' ')))
    n_var = len(vars)

    file.readline()
    line = file.readline()
    line = line.split(',')
    n_node = int(line[0].replace('N=',' ').replace(' ',''))
    n_edge = int(line[1].replace('E=',' ').replace(' ',''))
    file.close()

    out_dict = {}
    for var in var_dict:
        out_dict[var] = []
    # Read all time id
    for file_name in file_names:
        # Skip header and read data part
        file = open(file_name,'r')
        file.readline()
        file.readline()
        file.readline()
        data = file.read()
        file.close()

        # Process data string into numpy array of shape=(n_node, n_var)
        data = data.replace('\n',' ')
        data = list(filter(None, data.split(' ')))
        edge_index = data[n_var*n_node:n_var*n_node + 2 * n_edge]
        data = np.array(data[0:n_var*n_node], dtype=np.float32)
        data = data.reshape((n_node, n_var)).transpose()
        
        # Store to variable dict
        for var in var_dict:
            out_dict[var].append(np.expand_dims(data[vars.index(var_dict[var])], axis=-1))
        
    # Aggregate results from all time id.
    for var in var_dict:
        out_dict[var] = np.concatenate(out_dict[var], axis=-1)
    edge_index = np.array(edge_index, dtype = np.int32).reshape((n_edge, 2)).transpose() - 1
    # if out_dict['flowrate'] is not None:
    #     out_dict['flowrate'] = node_to_edge(out_dict['flowrate'], edge_index)
    return out_dict

def node_to_edge(node_attr, edge_index):
    return np.array([node_attr[i] for i in edge_index[1]])

def edge_to_node(edge_attr, edge_index):
    n_node = edge_index.max() + 1
    if len(edge_attr.shape) <=1:
        n_attr = 1
        node_attr = np.zeros(shape=(n_node,) , dtype=np.float32)
    else:
        n_attr = edge_attr.shape[1]
        node_attr = np.zeros(shape=(n_node, n_attr) , dtype=np.float32)
    for i in range(edge_index.shape[1]):
        node_attr[edge_index[1][i]] = edge_attr[i]
    # find root and assign root features
    is_child = np.isin(edge_index[0], edge_index[1])
    root = np.where(is_child == False)[0][0]
    node_attr[edge_index[0][root]] = edge_attr[root]
    return node_attr