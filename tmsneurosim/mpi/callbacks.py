from functools import partial, wraps
import gc
from typing import List

import numpy as np
from neuron import h
from sklearn.preprocessing import normalize

from tmsneurosim.mpi.data import MPIRecorder
from tmsneurosim.nrn.simulation import Backend as N
from tmsneurosim.nrn.simulation.e_field_simulation import EFieldSimulation


__cache__ = {}


def cached(fun, names):

    @wraps(fun)
    def ret_fun(*args, **kwargs):
        
        try:
            ret = tuple(__cache__[n] for n in names)
        except KeyError:
            ret = fun(*args, **kwargs)
        
        for n, v in zip(names, ret):
            __cache__[n] = v

        return ret

    return ret_fun


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


def min_max_normalize(v):
    return -1 + 2*((v - v.min()) / (v.max() - v.min()))


def euclidean_distance(section, origin):
    loc = np.array([section(0.5).x_xtra, section(0.5).y_xtra, section(0.5).z_xtra])
    return np.sqrt(np.square(loc-origin).sum())


class CallbackList:
    def __init__(self, callbacks):
        self.callbacks : List[Callback] = callbacks if callbacks is not None else []

    def init(self, n_cells, n_rotations, n_locations):
        for c in self.callbacks:
            c.init(n_cells, n_rotations, n_locations)

    def call_hook(
            self,
            hook,
            cell,
            state,
            waveform_type,
            position,
            transformed_e_field,
            threshold,
            idx):
        for c in self.callbacks:
            c.call_hook(
                hook,
                cell,
                state,
                waveform_type,
                position,
                transformed_e_field,
                threshold,
                idx)
        __cache__.clear()

    def close(self):
        for c in self:
            c.close()

    def __iter__(self):
        return iter(self.callbacks)


class Callback(MPIRecorder):

    __valid__ = {'post_threshold'}

    def __init__(self, directory, variables):
        super().__init__(directory, variables)

    def call_hook(
            self,
            hook,
            cell,
            state,
            waveform_type,
            position,
            transformed_e_field,
            threshold,
            idx):
        if hook not in self.__valid__:
            raise AttributeError(f'{hook} is invalid.')
        getattr(self, hook)(
            cell,
            state,
            waveform_type,
            position,
            transformed_e_field,
            threshold,
            idx)
        gc.collect()

    def post_threshold(self):
        pass


class ThresholdCallback(Callback):
    def __init__(self, directory, variables, terminals_only=True):
        super().__init__(directory, variables)
        self.terminals_only = terminals_only

    @partial(cached, names=['initiate_ind', 't_init'])
    def _determine_initiate_ind(
            self, cell, state, waveform_type, transformed_e_field, threshold
        ):

        v_thr = N.threshold

        simulation = EFieldSimulation(cell, waveform_type).attach()
        simulation.apply_e_field(transformed_e_field)

        if self.terminals_only:
            secs = cell.terminals()
        else:
            secs = cell.node + cell.unmyelin
        v_records = []

        for sec in secs:
            v_record = h.Vector()
            v_record.record(sec(0.5)._ref_v)
            v_records.append(v_record)

        simulation.simulate(threshold, init_state=state)
        v_rec = np.vstack([np.array(v) for v in v_records])

        sec_inds_t, t_inds_t = np.where(np.diff(np.signbit(v_rec-v_thr), axis=1))
        sec_inds = []
        t_inds = []

        len_ap = int(N.ap_dur / N.dt)
        
        for s_ind, t_ind in zip(sec_inds_t, t_inds_t):
            if np.all(v_rec[s_ind][t_ind+1:t_ind+len_ap] >= v_thr) or t_ind >= len_ap:
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
            t_init = (t_min + t_test) * N.dt
        else:
            initiate_ind = initiate_inds[0]
            dv = v_rec[initiate_ind][t_min+1] - v_rec[initiate_ind][t_min-1]
            t_pred = -v_rec[initiate_ind][t_min] / dv
            t_init = (t_min + t_pred) * N.dt
        
        return initiate_ind, t_init


class ThresholdDataRecorder(ThresholdCallback):
    def __init__(self,
                 directory,
                 terminals_only=True,
                 t_init=False):
        variables = [
            'soma_efield',
            'threshold',
            'initiate_ind',
            'position'
        ]
        super().__init__(directory, variables, terminals_only)
        self.t_init = t_init
        if t_init:
            self.make_record('t_init')

    def post_threshold(
            self,
            cell,
            state,
            waveform_type,
            position,
            transformed_e_field,
            threshold,
            idx) -> None:

        initiate_ind, t_init = self._determine_initiate_ind(
            cell, state, waveform_type, transformed_e_field, threshold
        )

        data = {
            'soma_efield': np.array([cell.soma[0].Ex_xtra,
                                     cell.soma[0].Ey_xtra,
                                     cell.soma[0].Ez_xtra]),
            'threshold': threshold,
            'initiate_ind': initiate_ind,
            'position': position
        }

        if self.t_init:
            data['t_init'] = t_init

        i, j, k = idx
        for key, value in data.items():
            self.save(key, i, j, k, value)


class ThresholdAmpScaleRecorder(ThresholdCallback):
    def __init__(self,
                 directory,
                 amp_scale_range: List[float],
                 record_soma=False,
                 record_apic=False):
        
        super().__init__(directory, variables=None, terminals_only=True)
        
        self.amp_scale_range = amp_scale_range
        self.record_soma = record_soma
        self.record_apic = record_apic

        for scale in self.amp_scale_range:
            self.make_record(f'v_axon_{scale:.2f}')
            if self.record_soma:
                self.make_record(f'v_soma_{scale:.2f}')
            if self.record_apic:
                self.make_record(f'v_apic_{scale:.2f}')

    def post_threshold(
            self,
            cell,
            state,
            waveform_type,
            position,
            transformed_e_field,
            threshold,
            idx) -> None:

        initiate_ind, _ = self._determine_initiate_ind(
            cell, state, waveform_type, transformed_e_field, threshold
        )

        i, j, k = idx

        simulation = EFieldSimulation(cell, waveform_type).attach()
        simulation.apply_e_field(transformed_e_field)

        terminal_sec = cell.terminals()[initiate_ind]

        v_rec_axon = h.Vector()
        v_rec_axon.record(terminal_sec(0.5)._ref_v)

        if self.record_soma:
            v_rec_soma = h.Vector()
            v_rec_soma.record(cell.soma[0](0.5)._ref_v)

        if self.record_apic:
            v_rec_apic = h.Vector()
            pick_from = cell.terminals(cell.apic)
            origin = np.array([terminal_sec(0.5).x_xtra,
                               terminal_sec(0.5).y_xtra,
                               terminal_sec(0.5).z_xtra])
            ds = [euclidean_distance(sec, origin) for sec in pick_from]
            apic_sec = pick_from[np.argmax(ds)]
            v_rec_apic.record(apic_sec(0.5)._ref_v)


        for scale in self.amp_scale_range:
            simulation.simulate(scale*threshold, init_state=state)
            self.save(f'v_axon_{scale:.2f}', i, j, k, np.array(v_rec_axon))
            if self.record_soma:
                self.save(f'v_soma_{scale:.2f}', i, j, k, np.array(v_rec_soma))
            if self.record_apic:
                self.save(f'v_apic_{scale:.2f}', i, j, k, np.array(v_rec_apic))


def make_nn_input(cell, neg=False):
    terminals = cell.terminals()
    
    ip = cell.terminal_efield_inner_prod()
    if neg:
        ip *= -1
    ip_n = np.expand_dims(min_max_normalize(ip), axis=0)
    
    af = cell.terminal_activating_funcs()
    if neg:
        af *= -1
    af_n = np.expand_dims(min_max_normalize(af), axis=0)

    es = np.array([t.es_xtra for t in terminals])
    if neg:
        es *= -1
    es_n = np.expand_dims(min_max_normalize(es), axis=0)
    
    soma = cell.soma[0]
    ef = np.array([[soma.Ex_xtra, soma.Ey_xtra, soma.Ez_xtra]])
    if neg:
        ef *= -1
    
    reorder = np.argsort(ip_n, axis=1)[:, ::-1]
    
    ip_r = np.expand_dims(np.take_along_axis(ip_n, reorder, axis=1)[:, :10], -1)
    af_r = np.expand_dims(np.take_along_axis(af_n, reorder, axis=1)[:, :10], -1)
    es_r = np.expand_dims(np.take_along_axis(es_n, reorder, axis=1)[:, :10], -1)
    
    ef_r = np.expand_dims(normalize(ef, axis=1), 1)
    ef_r = np.repeat(ef_r, 10, axis=1)
    
    ip_sum = np.expand_dims(np.sum(ip_n, axis=-1), (1, 2)) / 10
    ip_sum = np.repeat(ip_sum, 10, axis=1)
    
    x = np.concatenate([ip_r+af_r-es_r, ip_r, af_r, es_r, ef_r, ip_sum], axis=-1)
    
    return x, reorder[0, :10]