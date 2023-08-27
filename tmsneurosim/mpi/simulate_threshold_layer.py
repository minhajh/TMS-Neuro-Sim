import gc
from typing import Tuple, List, Iterator

import numpy as np
import simnibs
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from mpi4py import MPI
from neuron import h

from tmsneurosim.cortical_layer import CorticalLayer
from tmsneurosim.nrn.cells import NeuronCell
from tmsneurosim.nrn.simulation.e_field_simulation import EFieldSimulation
from tmsneurosim.nrn.simulation.simulation import WaveformType
from tmsneurosim.nrn.simulation import Backend as N

from tmsneurosim.mpi.data import (
    COMPUTE_TAG, FILE_TAG, MASTER_RANK, FILE_RANK
)
from tmsneurosim.mpi.callbacks import CallbackList, Callback


WHITE_MATTER_SURFACE = 1001
GRAY_MATTER_SURFACE = 1002

COMM = MPI.COMM_WORLD
RANK = COMM.rank
SIZE = COMM.size


def all_simulation_params(
        layer,
        cells,
        waveform_type,
        directions,
        positions, 
        rotation_count,
        rotation_step,
        azimuthal_rotation
    ):

    for i, cell in enumerate(cells):
        for j in range(rotation_count):
            rotations = azimuthal_rotation + j * rotation_step
            for k, (direction, position, rotation) in enumerate(zip(directions, positions, rotations)):
                idx = (i, j, k)
                yield idx, cell, waveform_type, direction, position, rotation, layer 


def simulate_combined_threshold_layer(
        layer: CorticalLayer,
        cells: List[NeuronCell],
        waveform_type: WaveformType,
        rotation_count: int,
        rotation_step: int,
        initial_rotation: int = None,
        callbacks: List[Callback] = None) -> simnibs.Msh:
    """
    Simulates the threshold of each neuron at each simulation element with all azimuthal rotations.
    :param layer: The cortical layer to place the neurons on.
    :param cells: The neurons to be used for the simulations.
    :param waveform_type: The waveform type to be used for the simulation.
    :param rotation_count: The number of azimuthal rotation per cell.
    :param rotation_step: The step width of the azimuthal rotations per cell.
    :param initial_rotation: The initial azimuthal rotation.
    :return: The layer with the results in the form of element fields.
    """

    if initial_rotation is None:
        if RANK == 0:
            azimuthal_rotation = np.random.rand(len(layer.elements)) * 360
        else:
            azimuthal_rotation = np.empty(len(layer.elements), dtype=np.float64)
        COMM.Bcast(azimuthal_rotation, 0)
    else:
        azimuthal_rotation = np.full(len(layer.elements),
                                     initial_rotation,
                                     dtype=np.float64)

    layer_meta_info = []
    for cell in cells:
        for i in range(rotation_count):
            layer_meta_info.append((cell, i))

    callbacks : CallbackList = CallbackList(callbacks)
    callbacks.init(len(cells), rotation_count, len(layer.elements))

    directions = layer.get_smoothed_normals()
    positions = layer.surface.elements_baricenters().value[layer.elements]

    total_sims = len(cells) * rotation_count * azimuthal_rotation.shape[0]

    all_params = all_simulation_params(layer,
                                       cells,
                                       waveform_type,
                                       directions,
                                       positions,
                                       rotation_count,
                                       rotation_step,
                                       azimuthal_rotation)
    
    layer_results = np.empty((len(cells),
                              rotation_count,
                              azimuthal_rotation.shape[0]),
                              dtype=np.float64)
    
    layer_tags = np.empty((len(cells),
                           rotation_count,
                           azimuthal_rotation.shape[0]),
                           dtype=np.int32)
    
    if RANK == MASTER_RANK:
        _master(total_sims, layer_results, layer_tags)
    else:
        _worker(all_params,
                callbacks=callbacks)

    COMM.barrier()

    if RANK == 0:
        layer.add_selected_elements_field(azimuthal_rotation, 'Initial_Rotation')
        layer.add_selected_elements_field(layer.get_smoothed_normals(), 'Normal')
        for i, cell in enumerate(cells):
            for j in range(rotation_count):
                res = layer_results[i, j]
                tag = layer_tags[i, j]
                layer.add_selected_elements_field(res, f'{cell.__class__.__name__}_{cell.morphology_id}__{j}')
                layer.add_selected_elements_field(tag, f'{cell.__class__.__name__}_{cell.morphology_id}_{j}_tags')


        if len(layer_meta_info) > 1:
            layer_thresholds = np.reshape(layer_results, (len(cells)*rotation_count, -1))
            layer.add_selected_elements_field(np.nanmedian(layer_thresholds, axis=0), 'Median_Threshold')
            layer.add_selected_elements_field(np.nanmean(layer_thresholds, axis=0), 'Mean_Threshold')
            layer.add_nearest_interpolation_field(np.nanmedian(layer_thresholds, axis=0), 'Median_Threshold_Visual')
            layer.add_nearest_interpolation_field(np.nanmean(layer_thresholds, axis=0), 'Mean_Threshold_Visual')

    return layer.surface


def _master(n_parameter_sets, threshes, tags):
    complete = 0
    deploy = np.empty(1, dtype='i')
    finish = np.array(-1, dtype='i')
    param_id = np.empty(3, dtype=np.int32)

    local_rec_thresh = np.empty(1, dtype=np.float64)
    local_rec_tag = np.empty(1, dtype=np.int32)

    pbar = tqdm(total=n_parameter_sets)

    # distribute initial jobs
    _distribute_initial_jobs(n_parameter_sets, deploy)

    # do rest
    while complete < n_parameter_sets:
        s = MPI.Status()
        COMM.Probe(status=s, tag=COMPUTE_TAG)
        COMM.Recv([param_id, MPI.INT32_T], source=s.source, tag=COMPUTE_TAG)
        COMM.Recv([local_rec_thresh, MPI.DOUBLE], source=s.source,
                  tag=COMPUTE_TAG)
        threshes[tuple(param_id)] = local_rec_thresh
        COMM.Recv([local_rec_tag, MPI.INT32_T], source=s.source,
                  tag=COMPUTE_TAG)
        tags[tuple(param_id)] = local_rec_tag
        if deploy < n_parameter_sets - 1:
            deploy += 1
            COMM.Send([deploy, MPI.INT], dest=s.source, tag=COMPUTE_TAG)
        else:
            COMM.Send([finish, MPI.INT], dest=s.source, tag=COMPUTE_TAG)
        pbar.update()
        complete += 1

    done = 0
    COMM.send(done, dest=FILE_RANK, tag=FILE_TAG)


def _worker(params,
            callbacks: CallbackList = None):
    
    deploy = np.empty(1, dtype='i')
    counter = -1
    gen = params
    param_id = np.empty(3, dtype=np.int32)

    local_rec_thresh = np.empty(1, dtype=np.float64)
    local_rec_tag = np.empty(1, dtype=np.int32)

    while True:
        COMM.Recv([deploy, MPI.INT], source=0, tag=COMPUTE_TAG)
        if deploy == -1:
            break
        r = None
        while counter != deploy:
            r = next(gen)
            counter += 1
        
        idx, cell, waveform_type, direction, position, rotation, layer = r
        threshold, tag = calculate_cell_threshold(cell,
                                                  waveform_type,
                                                  direction,
                                                  position,
                                                  rotation,
                                                  layer,
                                                  idx,
                                                  callbacks)
        gc.collect()
        local_rec_thresh[:] = threshold
        local_rec_tag[:] = tag
        param_id[:] = idx
        COMM.Send([param_id, MPI.INT32_T], dest=0, tag=COMPUTE_TAG)
        COMM.Send([local_rec_thresh, MPI.DOUBLE], dest=0,
                  tag=COMPUTE_TAG)
        COMM.Send([local_rec_tag, MPI.INT32_T], dest=0,
                  tag=COMPUTE_TAG)


def _distribute_initial_jobs(n: int, deploy: np.ndarray):
    available = SIZE - 2
    size = n if n < available else available
    available_ranks = list(range(SIZE))
    available_ranks.remove(MASTER_RANK)
    available_ranks.remove(FILE_RANK)

    for i in range(size):
        deploy[:] = i
        COMM.Send([deploy, MPI.INT], dest=available_ranks[i], tag=COMPUTE_TAG)

    if size < available:
        deploy[:] = -1
        for j in range(size, available):
            COMM.Send([deploy, MPI.INT], dest=available_ranks[j], tag=COMPUTE_TAG)
        deploy[:] = size - 1


def calculate_cell_threshold(
        cell: NeuronCell,
        waveform_type: WaveformType,
        direction: NDArray[np.float64],
        position: NDArray[np.float64],
        azimuthal_rotation: float,
        layer: CorticalLayer,
        idx: Tuple[int, int, int],
        callbacks: CallbackList = None) -> Tuple[float, int]:
    
    if np.any(np.isnan(direction)) or np.any(np.isnan(position)):
        return np.nan, 0
        
    cell.load()
    simulation = EFieldSimulation(cell, waveform_type)
    simulation.attach()
    
    segment_points_mm = cell.get_segment_coordinates() * 0.001
    azimuthal_rotation = Rotation.from_euler('z', azimuthal_rotation, degrees=True)
    rotation = rotation_from_vectors(cell.direction, direction)

    transformed_cell_coordinates = np.array(segment_points_mm) - segment_points_mm[0]
    transformed_cell_coordinates = rotation.apply(
        azimuthal_rotation.apply(transformed_cell_coordinates)) + position

    e_field_at_cell, tetrahedron_tags = layer.interpolate_scattered_cashed(
        transformed_cell_coordinates, 'E', out_fill=np.nan, get_tetrahedron_tags=True)
    if np.isnan(e_field_at_cell).any():
        tetrahedron_tags = np.append(tetrahedron_tags, [0])

    e_field_at_cell = np.nan_to_num(e_field_at_cell, nan=0)
    transformed_e_field = azimuthal_rotation.inv().apply(rotation.inv().apply(
        e_field_at_cell))

    simulation.apply_e_field(transformed_e_field)
    threshold = simulation.find_threshold_factor()
    
    callbacks.call_hook(
        'post_threshold',
        cell,
        waveform_type,
        position,
        transformed_e_field,
        threshold,
        idx
    )
            
    simulation.detach()
    cell.unload()
    gc.collect()
    return threshold, int(''.join(map(str, np.unique(tetrahedron_tags)))[::-1])


def rotation_from_vectors(
        vec1: NDArray[np.float64],
        vec2: NDArray[np.float64]) -> Rotation:
    """ Creates a rotation instance that rotates vector 1 into the direction of vector 2

    :param vec1: The original direction
    :param vec2: The final direction
    :return: A rotation instance that rotates vector 1 into the direction of vector 2
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return Rotation.identity()
    vec1_normalized = (vec1 / np.linalg.norm(vec1)).reshape(3)
    vec2_normalized = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(vec1_normalized, vec2_normalized)
    rotation = Rotation.identity()
    if any(v):
        c = np.dot(vec1_normalized, vec2_normalized)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation = Rotation.from_matrix(np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2)))
    return rotation


def euclidean_distance(section, origin):
    loc = np.array([section(0.5).x_xtra, section(0.5).y_xtra, section(0.5).z_xtra])
    return np.sqrt(np.square(loc-origin).sum())

