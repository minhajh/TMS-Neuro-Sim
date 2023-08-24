import os

from mpi4py import MPI
import numpy as np

COMPUTE_TAG = 0
END_TAG = 99
FILE_TAG = 11

MASTER_RANK = 0
FILE_RANK = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
                    s = (n_cells, n_rotations, n_locations, *shape)
                    fp = np.memmap(
                        self.directory+'/'+var,
                        dtype=dtype,
                        mode='w+',
                        shape=s)
                    del fp
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
                d = [var, tuple(data.shape), data.dtype]
                comm.send(d, dest=FILE_RANK, tag=FILE_TAG)
                fp = np.memmap(
                    self.directory+'/'+var,
                    dtype=data.dtype,
                    mode='r+',
                    shape=s)
                self.records[var] = fp
            self.records[var][i, j, k, :] = data

    def close(self):
        for k in self.records.keys():
            del self.records[k]
        comm.Barrier()


class MPIRecorder:
    def __init__(self, directory):
        self.directory = directory
        if rank == 0:
            os.makedirs(directory, exist_ok=True)
        self.amode = MPI.MODE_WRONLY|MPI.MODE_CREATE