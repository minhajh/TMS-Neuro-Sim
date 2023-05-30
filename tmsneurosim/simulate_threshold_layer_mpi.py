import gc
import os
from typing import Tuple, List
import pickle
import random

import numpy as np
import simnibs
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from mpi4py import MPI
from neuron import h

import matplotlib.pyplot as plt

from tmsneurosim.cortical_layer import CorticalLayer
from tmsneurosim.nrn.cells import NeuronCell
from tmsneurosim.nrn.simulation.e_field_simulation import EFieldSimulation
from tmsneurosim.nrn.simulation.simulation import WaveformType
from tmsneurosim.nrn.simulation import Backend as N
from tmsneurosim.nrn.cells.cell_modification_parameters.cell_modification_parameters import (
    CellModificationParameters, AxonModificationMode)

WHITE_MATTER_SURFACE = 1001
GRAY_MATTER_SURFACE = 1002

COMPUTE_TAG = 0
END_TAG = 99

COMM = MPI.COMM_WORLD
RANK = COMM.rank
SIZE = COMM.size


def get_branch_from_terminal(cell, terminal_sec, terminate_before_soma=False):
    branch = []
    sref = h.SectionRef(terminal_sec)
    branch.append(sref.sec)
    while sref.sec is not cell.soma[0]:
        p = sref.parent
        sref = h.SectionRef(p)
        if terminate_before_soma and sref.sec is cell.soma[0]:
            break
        branch.append(sref.sec)
    return branch

def get_branch_to_branch(cell, terminal_sec, adjacency):
    branch = []
    sref = h.SectionRef(terminal_sec)
    secs = cell.node + cell.unmyelin
    branch.append(sref.sec)
    while sref.sec is not cell.soma[0]:
        p = sref.parent
        sref = h.SectionRef(p)
        branch.append(sref.sec)
        if sref.sec in secs:
            if np.count_nonzero(adjacency[secs.index(sref.sec)]) > 2:
                break
    return branch

def full_branch(cell, terminal_sec, adjacency):
    main_branch = get_branch_from_terminal(cell, terminal_sec)
    aux = []
    for t_sec in cell.terminals():
        aux_branch = get_branch_to_branch(cell, t_sec, adjacency)
        if any([sec in main_branch for sec in aux_branch]):
            aux += aux_branch
    return main_branch + aux


def all_simulation_params(layer, cells, waveform_type, directions, positions, 
                          rotation_count, rotation_step, azimuthal_rotation):

    for i, cell in enumerate(cells):
        for j in range(rotation_count):
            rotations = azimuthal_rotation + j * rotation_step
            for k, (direction, position, rotation) in enumerate(zip(directions, positions, rotations)):
                idx = (i, j, k)
                yield idx, cell, waveform_type, direction, position, rotation, layer 


def simulate_combined_threshold_layer(layer: CorticalLayer,
                                      cells: List[NeuronCell],
                                      waveform_type: WaveformType,
                                      rotation_count: int,
                                      rotation_step: int,
                                      initial_rotation: int = None,
                                      record=False,
                                      record_all=False,
                                      directory=None,
                                      record_v=True,
                                      amp_scale_range=None,
                                      record_disconnected=False,
                                      random_disconnected_axon=False,
                                      random_disconnected_dend=False,
                                      pick_furthest_dend=False,
                                      include_basal=True,
                                      apic_branch_diam_scale=2.0) -> simnibs.Msh:
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
        azimuthal_rotation = np.full(len(layer.elements), initial_rotation, dtype=np.float64)

    layer_meta_info = []
    for cell in cells:
        for i in range(rotation_count):
            layer_meta_info.append((cell, i))


    directions = layer.get_smoothed_normals()
    positions = layer.surface.elements_baricenters().value[layer.elements]

    total_sims = len(cells) * rotation_count * azimuthal_rotation.shape[0]

    all_params = all_simulation_params(layer, cells, waveform_type, directions, positions,
                                       rotation_count, rotation_step, azimuthal_rotation)
    
    layer_results = np.empty((len(cells), rotation_count, azimuthal_rotation.shape[0]), dtype=np.float64)
    layer_tags = np.empty((len(cells), rotation_count, azimuthal_rotation.shape[0]), dtype=np.int32)
    
    if SIZE == 1:
        run_all(all_params,
                layer_results,
                layer_tags,
                record=record,
                record_all=record_all,
                record_v=record_v,
                directory=directory,
                amp_scale_range=amp_scale_range,
                record_disconnected=record_disconnected,
                total=total_sims,
                random_disconnected_axon=random_disconnected_axon,
                random_disconnected_dend=random_disconnected_dend,
                include_basal=include_basal,
                pick_furthest_dend=pick_furthest_dend)

    else:
        if RANK == 0:
            _master(total_sims, layer_results, layer_tags)
        else:
            _worker(all_params,
                    record=record,
                    record_all=record_all,
                    record_v=record_v,
                    directory=directory,
                    amp_scale_range=amp_scale_range,
                    record_disconnected=record_disconnected,
                    random_disconnected_axon=random_disconnected_axon,
                    random_disconnected_dend=random_disconnected_dend,
                    include_basal=include_basal,
                    pick_furthest_dend=pick_furthest_dend,
                    apic_branch_diam_scale=apic_branch_diam_scale)

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


def run_all(params,
            layer_results,
            layer_tags,
            record=False,
            record_all=False,
            record_v=True,
            directory=None,
            amp_scale_range=None,
            record_disconnected=False,
            random_disconnected_axon=False,
            random_disconnected_dend=False,
            include_basal=True,
            pick_furthest_dend=True,
            total=None):
    
    for r in tqdm(params, total=total):
        idx, cell, waveform_type, direction, position, rotation, layer = r
        threshold, tag = calculate_cell_threshold(cell,
                                                  waveform_type,
                                                  direction,
                                                  position,
                                                  rotation,
                                                  layer,
                                                  record=record,
                                                  record_all=record_all,
                                                  idx=idx,
                                                  directory=directory,
                                                  record_v=record_v,
                                                  amp_scale_range=amp_scale_range,
                                                  record_disconnected=record_disconnected,
                                                  random_disconnected_axon=random_disconnected_axon,
                                                  random_disconnected_dend=random_disconnected_dend,
                                                  include_basal=include_basal,
                                                  pick_furthest_dend=pick_furthest_dend)
        gc.collect()
        layer_results[tuple(idx)] = threshold
        layer_tags[tuple(idx)] = tag


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
        COMM.Probe(status=s)
        COMM.Recv([param_id, MPI.INT32_T], source=s.source)
        COMM.Recv([local_rec_thresh, MPI.DOUBLE], source=s.source,
                  tag=COMPUTE_TAG)
        threshes[tuple(param_id)] = local_rec_thresh
        COMM.Recv([local_rec_tag, MPI.INT32_T], source=s.source,
                  tag=COMPUTE_TAG)
        tags[tuple(param_id)] = local_rec_tag
        if deploy < n_parameter_sets - 1:
            deploy += 1
            COMM.Send([deploy, MPI.INT], dest=s.source)
        else:
            COMM.Send([finish, MPI.INT], dest=s.source)
        pbar.update()
        complete += 1


def _worker(params,
            record=False,
            record_all=False,
            record_v=True,
            directory=None,
            amp_scale_range=None,
            record_disconnected=False,
            random_disconnected_axon=False,
            random_disconnected_dend=False,
            include_basal=True,
            pick_furthest_dend=True,
            apic_branch_diam_scale=2.0):
    
    deploy = np.empty(1, dtype='i')
    counter = -1
    gen = params
    param_id = np.empty(3, dtype=np.int32)

    local_rec_thresh = np.empty(1, dtype=np.float64)
    local_rec_tag = np.empty(1, dtype=np.int32)

    while True:
        COMM.Recv([deploy, MPI.INT], source=0)
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
                                                  record=record,
                                                  record_all=record_all,
                                                  idx=idx,
                                                  directory=directory,
                                                  record_v=record_v,
                                                  amp_scale_range=amp_scale_range,
                                                  record_disconnected=record_disconnected,
                                                  random_disconnected_axon=random_disconnected_axon,
                                                  random_disconnected_dend=random_disconnected_dend,
                                                  include_basal=include_basal,
                                                  pick_furthest_dend=pick_furthest_dend,
                                                  apic_branch_diam_scale=apic_branch_diam_scale)
        gc.collect()
        local_rec_thresh[:] = threshold
        local_rec_tag[:] = tag
        param_id[:] = idx
        COMM.Send([param_id, MPI.INT32_T], dest=0)
        COMM.Send([local_rec_thresh, MPI.DOUBLE], dest=0,
                  tag=COMPUTE_TAG)
        COMM.Send([local_rec_tag, MPI.INT32_T], dest=0,
                  tag=COMPUTE_TAG)


def _distribute_initial_jobs(n: int, deploy: np.ndarray):
    available = SIZE - 1
    size = n if n < available else available
    available_ranks = list(range(SIZE))
    available_ranks.remove(0)
    for i in range(size):
        deploy[:] = i
        COMM.Send([deploy, MPI.INT], dest=available_ranks[i])

    if size < available:
        deploy[:] = -1
        for j in range(size, available):
            COMM.Send([deploy, MPI.INT], dest=available_ranks[j])
        deploy[:] = size - 1


def calculate_cell_threshold(cell: NeuronCell,
                             waveform_type: WaveformType,
                             direction: NDArray[np.float64],
                             position: NDArray[np.float64],
                             azimuthal_rotation: float,
                             layer: CorticalLayer,
                             record=False,
                             record_all=False,
                             record_v=True,
                             idx=None,
                             directory: str = None,
                             amp_scale_range=None,
                             record_disconnected=False,
                             random_disconnected_axon=False,
                             random_disconnected_dend=False,
                             include_basal=True,
                             pick_furthest_dend=True,
                             apic_branch_diam_scale=2.0) -> Tuple[float, int]:
    
    if np.any(np.isnan(direction)) or np.any(np.isnan(position)):
        return np.nan, 0
    cell.load()
    simulation = EFieldSimulation(cell, waveform_type)
    simulation.attach()
    segment_points_mm = cell.get_segment_coordinates() * 0.001
    azimuthal_rotation = Rotation.from_euler('z', azimuthal_rotation, degrees=True)
    rotation = rotation_from_vectors(cell.direction, direction)

    transformed_cell_coordinates = np.array(segment_points_mm) - segment_points_mm[0]
    transformed_cell_coordinates = rotation.apply(azimuthal_rotation.apply(transformed_cell_coordinates)) + position

    e_field_at_cell, tetrahedron_tags = layer.interpolate_scattered_cashed(
        transformed_cell_coordinates, 'E', out_fill=np.nan, get_tetrahedron_tags=True)
    if np.isnan(e_field_at_cell).any():
        tetrahedron_tags = np.append(tetrahedron_tags, [0])

    e_field_at_cell = np.nan_to_num(e_field_at_cell, nan=0)
    transformed_e_field = azimuthal_rotation.inv().apply(rotation.inv().apply(e_field_at_cell))

    simulation.apply_e_field(transformed_e_field)
    threshold = simulation.find_threshold_factor()
    
    v_thresh = N.threshold

    if record:
        if not record_all:
            secs = cell.node + cell.unmyelin
        else:
            secs = cell.all

        v_records = []
        # e_records = []

        soma_record = h.Vector()
        soma_record.record(cell.soma[0](0.5)._ref_v)

        axon_0 = h.Vector()
        axon_0.record(cell.axon[0](0.5)._ref_v)

        es = [sec(0.5).es_xtra for sec in secs]

        for sec in secs:
            v_record = h.Vector()
            v_record.record(sec(0.5)._ref_v)
            v_records.append(v_record)
            #e_record = h.Vector()
            #e_record.record(sec(0.5)._ref_e_extracellular)
            #e_records.append(e_record)

        simulation.simulate(threshold, reinit=True)
        v_rec = np.vstack([np.array(v) for v in v_records])
        # e_rec = np.vstack([np.array(e) for e in e_records])

        sec_inds_t, t_inds_t = np.where(np.diff(np.signbit(v_rec-v_thresh), axis=1))
        sec_inds = []
        t_inds = []

        len_ap = int(N.ap_dur / N.dt)
        
        for s_ind, t_ind in zip(sec_inds_t, t_inds_t):
            if np.all(v_rec[s_ind][t_ind+1:t_ind+len_ap] >= v_thresh) or t_ind >= len_ap:
                sec_inds.append(s_ind)
                t_inds.append(t_ind)
        sec_inds = np.array(sec_inds)
        t_inds = np.array(t_inds)

        t_min_loc = np.where(t_inds==t_inds.min())[0]
        initiate_inds = sec_inds[t_min_loc]
        t_min = t_inds.min()

        if len(initiate_inds) > 1:
            initiate_ind = None
            t_test = np.inf
            for iind in initiate_inds:
                dv = v_rec[iind][t_min+1] - v_rec[iind][t_min-1]
                t_pred = -v_rec[iind][t_min] / dv
                if t_pred < t_test:
                    initiate_ind = iind
                    t_test = t_pred
            t_init = (t_min + t_test) * 0.005
        else:
            initiate_ind = initiate_inds[0]
            dv = v_rec[initiate_ind][t_min+1] - v_rec[initiate_ind][t_min-1]
            t_pred = -v_rec[initiate_ind][t_min] / dv
            t_init = (t_min + t_pred) * 0.005

        data = {
            'soma_rec': np.array(soma_record),
            'axon_0': np.array(axon_0),
            'es': np.array(es),
            'threshold': threshold,
            'initiate_ind': initiate_ind,
            't_init': t_init
        }

        if record_v:
            data['v_rec_thresh'] = v_rec

        i, j, k = idx
        save_dir = f"{directory}/{cell.__class__.__name__}/{cell.morphology_id:02d}/{j:02d}/{k:05d}/"
        os.makedirs(save_dir, exist_ok=True)
        for key, value in data.items():
            np.save(save_dir+key, value)

        if amp_scale_range is not None:
            for scale in amp_scale_range:
                simulation.simulate(threshold*scale, reinit=True)
                v_rec = np.vstack([np.array(v) for v in v_records])
                # e_rec = np.vstack([np.array(e) for e in e_records])
                v_name = f'v_rec_thresh_{scale}'
                # e_name = f'e_rec_thresh_{scale}'
                np.save(save_dir+v_name, v_rec)
                # np.save(save_dir+e_name, e_rec)

        if record_disconnected:
            
            # -- branch to soma --

            v_records.clear()
            cell.unload()
            cell.load()
            simulation = EFieldSimulation(cell, waveform_type)
            simulation.apply_e_field(transformed_e_field)

            if not record_all:
                secs = cell.node + cell.unmyelin
            else:
                secs = cell.all

            init_sec = secs[initiate_ind]

            if include_basal:
                dendritic_comps = cell.apic + cell.dend
            else:
                dendritic_comps = cell.apic

            dendritic_terminals = cell.terminals(dendritic_comps)
            
            soma = cell.soma[0](0.5)
            e_field_soma = np.array([soma.Ex_xtra, soma.Ey_xtra, soma.Ez_xtra])
            np.save(save_dir+'e_field_soma', e_field_soma)
            
            es_dend = [sec(0.5).es_xtra for sec in dendritic_terminals]
            top_d = np.argmax(es_dend)

            if random_disconnected_axon:
                terminal_sec = random.choice(cell.terminals())
            else:
                terminal_sec = init_sec

            branch = get_branch_from_terminal(cell, terminal_sec)

            """
            apic_branches = []
            for d in top_d[:5]:
                apic_branch = get_branch_from_terminal(cell, cell.terminals(cell.apic)[d])
                apic_branches += apic_branch
            """

            if random_disconnected_dend:
                terminal_sec = random.choice(dendritic_terminals)
            else:
                if not pick_furthest_dend:
                    terminal_sec = dendritic_terminals[top_d]
                else:
                    pick_from = dendritic_terminals
                    origin = np.array([init_sec(0.5).x_xtra, init_sec(0.5).y_xtra, init_sec(0.5).z_xtra])
                    ds = [euclidean_distance(sec, origin) for sec in pick_from]
                    terminal_sec = pick_from[np.argmax(ds)]

            apic_branch = get_branch_from_terminal(cell, terminal_sec, terminate_before_soma=True)

            for sec in apic_branch:
                for seg in sec:
                    seg.diam *= apic_branch_diam_scale

            cell.unload_except(branch + apic_branch)

            simplified = (branch + apic_branch[::-1])[::-1]

            es_unbranched = np.array([sec(0.5).es_xtra for sec in simplified])
            dist_unbranched = np.array([h.distance(simplified[0](0.5), s(0.5))
                                        for s in simplified])
            np.save(save_dir+'es_unbranched', es_unbranched)
            np.save(save_dir+'dist_unbranched', dist_unbranched)

            e_field_simp = []
            for sec in simplified:
                e_field_simp.append([sec(0.5).Ex_xtra, sec(0.5).Ey_xtra, sec(0.5).Ez_xtra])
            e_field_simp = np.array(e_field_simp)

            np.save(save_dir+'e_field_simplified', e_field_simp)

            simulation.attach()

            """
            if not record_all:
                secs = cell.node + cell.unmyelin
            else:
                secs = cell.all

            for sec in secs:
                v_record = h.Vector()
                v_record.record(sec(0.5)._ref_v)
                v_records.append(v_record)

            v_record_init_sec = h.Vector().record(init_sec(0.5)._ref_v)
            """

            threshold_reduced = simulation.find_threshold_factor()

            # v_rec = np.vstack([np.array(v) for v in v_records])

            np.save(save_dir+'threshold_reduced', threshold_reduced)
            # np.save(save_dir+'v_branch_to_soma_init_sec', v_record_init_sec.as_numpy())

            # -- branch to branch --
            """
            v_records.clear()
            cell.unload()
            cell.load()
            simulation = EFieldSimulation(cell, waveform_type)
            simulation.attach(spike_recording=False)
            simulation.apply_e_field(transformed_e_field)

            if not record_all:
                secs = cell.node + cell.unmyelin
            else:
                secs = cell.all

            init_sec = secs[initiate_ind]

            branch = get_branch_to_branch(cell, init_sec, cell.adjacency())

            cell.unload_except(branch)

            if not record_all:
                secs = cell.node + cell.unmyelin
            else:
                secs = cell.all

            for sec in secs:
                v_record = h.Vector()
                v_record.record(sec(0.5)._ref_v)
                v_records.append(v_record)

            v_record_init_sec = h.Vector().record(init_sec(0.5)._ref_v)

            simulation.simulate(threshold, reinit=True)
            v_rec = np.vstack([np.array(v) for v in v_records])

            np.save(save_dir+'v_branch_to_branch', v_rec)
            np.save(save_dir+'v_branch_to_branch_init_sec', v_record_init_sec.as_numpy())
            """

    simulation.detach()
    return threshold, int(''.join(map(str, np.unique(tetrahedron_tags)))[::-1])


def rotation_from_vectors(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> Rotation:
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

