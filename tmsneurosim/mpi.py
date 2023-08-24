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
        self.records = {}

    def init(self, n_cells, n_rotations, n_locations):

        self.n_cells = n_cells
        self.n_rotations = n_rotations
        self.n_locations = n_locations

        if self.variables is not None:
            if rank == MASTER_RANK:
                for var, dtype, shape in self.variables:
                    self.make(var, dtype, shape)
            comm.Barrier()
            for var, dtype, shape in self.variables:
                s = (n_cells, n_rotations, n_locations, *shape)
                fp = np.memmap(
                    self.directory+'/'+var,
                    dtype=dtype,
                    mode='r+',
                    shape=s)
                self.records[var] = fp
            comm.Barrier()

    def save(self, var, i, j, k, data):
        data = np.atleast_1d(data)
        try:
            self.records[var][i, j, k, :] = data
        except KeyError:
            s = (self.n_cells, self.n_rotations, self.n_locations, *data.shape)
            if os.path.exists(self.directory+'/'+var):
                fp = np.memmap(
                    self.directory+'/'+var,
                    dtype=data.dtype,
                    mode='r+',
                    shape=s)
                self.records[var] = fp
            else:
                d = [var, data.dtype, tuple(data.shape)]
                comm.send(d, dest=FILE_RANK, tag=FILE_TAG)
                _ = comm.recv(source=FILE_RANK, tag=FILE_TAG)
                fp = np.memmap(
                    self.directory+'/'+var,
                    dtype=data.dtype,
                    mode='r+',
                    shape=s)
                self.records[var] = fp
            self.records[var][i, j, k, :] = data

    def close(self):
        keys = list(self.records.keys())
        for k in keys:
            del self.records[k]
        comm.Barrier()

    def make(self, var, dtype, shape):
        s = (self.n_cells, self.n_rotations, self.n_locations, *shape)
        fp = np.memmap(
            self.directory+'/'+var,
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
    def __init__(self, directory):
        self.directory = directory
        if rank == 0:
            os.makedirs(directory, exist_ok=True)
        self.amode = MPI.MODE_WRONLY|MPI.MODE_CREATE