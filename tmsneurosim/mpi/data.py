import os
import sys
import pickle

from mpi4py import MPI
import numpy as np

COMPUTE_TAG = 0
END_TAG = 99
FILE_TAG = 11

MASTER_RANK = 0
FILE_RANK = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

IS_COMPUTE_RANK = (rank != MASTER_RANK)

record_comm = comm.Split(color=int(IS_COMPUTE_RANK))


def print_immediately(*txt):
    print(*txt)
    sys.stdout.flush()


class Recorder:
    def __init__(self, directory, variables=None):
        if rank == 0:
            os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.variables = variables
        self.n_cells : int = None
        self.n_rotations : int = None
        self.n_locations : int = None

    def init(self, n_cells, n_rotations, n_locations):

        self.n_cells = n_cells
        self.n_rotations = n_rotations
        self.n_locations = n_locations

        if self.variables is not None:
            if rank == MASTER_RANK:
                for var, dtype, shape in self.variables:
                    self.make(var, dtype, shape)

        comm.Barrier()

    def save(self, var, i, j, k, data):

        data = np.atleast_1d(data).flatten()

        if os.path.exists(self.directory+'/'+var):
            fp = np.memmap(
                self.directory+'/'+var,
                dtype=data.dtype,
                mode='r+',
                shape=data.shape,
                offset=self.offset(data, i, j, k))
        else:
            d = [var, data.dtype, tuple(data.shape)]
            comm.send(d, dest=FILE_RANK, tag=FILE_TAG)
            _ = comm.recv(source=FILE_RANK, tag=FILE_TAG)
            while not os.path.exists(self.directory+'/'+var):
                pass
            fp = np.memmap(
                self.directory+'/'+var,
                dtype=data.dtype,
                mode='r+',
                shape=data.shape,
                offset=self.offset(data, i, j, k))
            
        fp[:] = data
        fp.flush()
        del fp

    def close(self):
        comm.Barrier()

    def offset(self, data, i, j, k):
        B = data.dtype.itemsize
        dl = np.prod(data.shape)
        N = np.array([self.n_cells, self.n_rotations, self.n_locations, dl])
        n = [i, j, k, 0]
        offset = 0
        for ii in range(len(n)):
            offset += n[ii]*np.prod(N[ii+1:])
        offset = B * offset
        return offset

    def make(self, var, dtype, shape):
        path = self.directory+'/'+var
        if not os.path.exists(path):
            s = (self.n_cells, self.n_rotations, self.n_locations, *shape)
            fp = np.memmap(path,
                           dtype=dtype,
                           mode='w+',
                           shape=s)
            del fp
            meta = {'shape':s, 'dtype':dtype}
            with open(f'{self.directory}/{var}.meta', 'wb') as f:
                pickle.dump(meta, f)

    @staticmethod
    def load(directory, var):
        path = f'{directory}/{var}'
        mpath = f'{directory}/{var}.meta'
        with open(mpath, 'rb') as f:
            meta = pickle.load(f)
        mm = np.memmap(path, dtype=meta['dtype'], mode='r', shape=meta['shape'])
        return mm


class MPIRecorder:
    
    def __init__(self, directory, variables):
        self.directory = directory
        
        if rank == 0:
            os.makedirs(directory, exist_ok=True)
            
        comm.Barrier()
        
        self.amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
        self.var_names = variables
        self.variables = {}

        if IS_COMPUTE_RANK:
            for var in variables:
                fh = MPI.File.Open(record_comm, f'{directory}/{var}', self.amode)
                self.variables[var] = fh
                
        self.n_cells : int = None
        self.n_rotations : int = None
        self.n_locations : int = None
        
    def init(self, n_cells, n_rotations, n_locations):

        self.n_cells = n_cells
        self.n_rotations = n_rotations
        self.n_locations = n_locations
        
        comm.Barrier()
                
    def offset(self, data, i, j, k):
        B = data.dtype.itemsize
        dl = np.prod(data.shape)
        N = np.array([self.n_cells, self.n_rotations, self.n_locations, dl])
        n = [i, j, k, 0]
        offset = 0
        for ii in range(len(n)):
            offset += n[ii]*np.prod(N[ii+1:])
        offset = B * offset
        return offset
        
    def save(self, var, i, j, k, data, dtype=None):
        data = np.atleast_1d(data)
        shape = data.shape
        data = np.atleast_1d(np.asarray(data, dtype=dtype)).flatten()

        if not os.path.exists(f'{self.directory}/{var}.meta'):
            s = (self.n_cells, self.n_rotations, self.n_locations, *shape)
            meta = {'shape':s, 'dtype':data.dtype}
            with open(f'{self.directory}/{var}.meta', 'wb') as f:
                pickle.dump(meta, f)
                
        offset = self.offset(data, i, j, k)
        self.variables[var].Write_at(offset, data)
        
    def close(self):
        if IS_COMPUTE_RANK:
            for var in self.var_names:
                self.variables[var].Close()
        comm.Barrier()
        
        
def load_data(directory, var):
    path = f'{directory}/{var}'
    mpath = f'{directory}/{var}.meta'
    with open(mpath, 'rb') as f:
        meta = pickle.load(f)
    mm = np.memmap(path, dtype=meta['dtype'], mode='r', shape=meta['shape'])
    return mm