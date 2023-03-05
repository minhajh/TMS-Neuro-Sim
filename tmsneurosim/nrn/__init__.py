import pathlib
import subprocess

import neuron
import shutil
from mpi4py import MPI

from ..nrn import __file__


COMM = MPI.COMM_WORLD
mechpath = pathlib.Path(__file__).parent.joinpath('mechanisms')

if not neuron.load_mechanisms(str(mechpath.resolve()), warn_if_already_loaded=False):
    COMM.barrier()

    if COMM.rank == 0:
        print('NEURON compile mechanisms (Only on first load)')
        n = subprocess.Popen([shutil.which('nrnivmodl')], cwd=mechpath,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        n.wait()

    COMM.barrier()
    neuron.load_mechanisms(str(mechpath.resolve()))
    COMM.barrier()

    if COMM.rank == 0:
        print('NEURON mechanisms loaded')

neuron.h.load_file('stdrun.hoc')
