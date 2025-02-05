import pathlib
from enum import Enum

import numpy as np
from neuron import h
from scipy.io import loadmat
from scipy.interpolate import interp1d

import tmsneurosim
from tmsneurosim.nrn.cells import NeuronCell
from tmsneurosim.nrn.simulation import Backend as N


class WaveformType(Enum):
    MONOPHASIC = 1
    BIPHASIC = 2
    HALFSINE = 3


WAVE_TYPE_STR = {
    WaveformType.MONOPHASIC:'m',
    WaveformType.BIPHASIC:'b',
    WaveformType.HALFSINE:'h'
}


class Simulation:
    """ Wrapper to set up, modify and execute a NEURON simulation of a single cell.

     Attributes:
            neuron_cell (NeuronCell): The cell that is supposed to be simulated
            stimulation_delay (float): Initial delay before the activation waveform is applied in s
            simulation_temperature (float): Temperature for the simulation in degree Celsius
            simulation_time_step (float): The time step used for the simulation in ms.
            simulation_duration (float): The duration of the simulation in ms.
            waveform ([float]): The amplitude values of the waveform used.
            waveform_time ([float]): The time values of the waveform used.
    """
    INITIAL_VOLTAGE = -70

    def __init__(self, neuron_cell: NeuronCell, waveform_type: WaveformType):
        """
        Initializes the simulation with the transmitted neuron cell and waveform type.
        :param neuron_cell: The neuron that is supposed to be used in the NEURON simulation.
        :param waveform_type: The waveform type that is supposed to be used in the NEURON simulation
        """
        self.neuron_cell = neuron_cell
        self._action_potentials = h.Vector()
        self._action_potentials_recording_ids = h.Vector()

        self.stimulation_delay = N.delay
        self.simulation_temperature = N.temp
        self.simulation_time_step = N.dt
        self.simulation_duration = N.tstop
        self.simulation_threshold = N.threshold
        self.waveform, self.waveform_time = self._load_waveform(waveform_type)

        self.init_handler = None
        self.init_state = None
        self.attached = False

        self.netcons = []


    def attach(self, spike_recording=True, steady_state=False):
        """
        Attaches spike recording to the neuron and connects the simulation initialization
        methode to the global NEURON space.
        """
        if spike_recording:
            self._init_spike_recording()

        if steady_state:
            self.init_handler = h.FInitializeHandler(2, self._post_finitialize)

        self.attached = True
        
        return self


    def detach(self):
        """ Removes the spike recording from the neuron and disconnects the initialization methode.
        """
        if self.attached:
            for net in self.netcons:
                net.record()
            self.netcons.clear()
            del self.init_handler
            self.init_handler = None

            self.attached = False
           
        return self


    def _post_finitialize(self):
        """ Initialization methode to unsure a steady state before the actual simulation is started.
        """
        temp_dt = h.dt

        h.t = -1e11
        h.dt = 1e9

        while h.t < - h.dt:
            h.fadvance()

        h.dt = temp_dt
        h.t = 0
        
        h.fcurrent()
        h.frecord_init()


    def _init_spike_recording(self):
        """Initializes spike recording for every segment of the neuron.
        """
        self.netcons = []
        self.v_recs = []
        for i, section in enumerate(self.neuron_cell.all):
            if N.validate:
                if section not in self.neuron_cell.myelin:
                    v = h.Vector()
                    v.record(section(0.5)._ref_v)
                    self.v_recs.append(v)
            for segment in section:
                recording_netcon = h.NetCon(segment._ref_v, None, sec=section)
                recording_netcon.threshold = self.simulation_threshold
                recording_netcon.delay = 0
                recording_netcon.record(self._action_potentials, self._action_potentials_recording_ids, i)
                self.netcons.append(recording_netcon)


    def validate_is_active(self):
        v_rec = np.vstack([np.array(v) for v in self.v_recs])
        sec_inds_t, t_inds_t = np.where(np.diff(np.signbit(v_rec-self.simulation_threshold), axis=1))
        for s_ind, t_ind in zip(sec_inds_t, t_inds_t):
            if np.all(v_rec[s_ind][t_ind+1:t_ind+50] >= self.simulation_threshold) or t_ind >= 50:
                return True
        return False


    def _load_waveform_old(self, waveform_type: WaveformType):
        """Loads the submitted waveform and modifies it to fit the simulation settings.
        """
        tms_waves = loadmat(
            str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath('coil_recordings/TMSwaves.mat').absolute()))

        recorded_time = tms_waves['tm'].ravel()
        recorded_e_field_magnitude = tms_waves['Erec_m']

        if waveform_type is WaveformType.BIPHASIC:
            recorded_e_field_magnitude = tms_waves['Erec_b']

        sample_factor = int(self.simulation_time_step / np.mean(np.diff(recorded_time)))
        if sample_factor < 1:
            sample_factor = 1

        simulation_time = recorded_time[::sample_factor]
        simulation_e_field_magnitude = np.append(recorded_e_field_magnitude[::sample_factor], 0)

        if self.stimulation_delay >= self.simulation_time_step:
            simulation_time = np.concatenate(
                (np.array([0, self.stimulation_delay - self.simulation_time_step]),
                 simulation_time + self.stimulation_delay))
            simulation_e_field_magnitude = np.concatenate(
                (np.array([0, 0]), np.append(recorded_e_field_magnitude[::sample_factor], 0)))

        simulation_time = np.append(np.concatenate(
            (simulation_time, np.arange(simulation_time[-1] + self.simulation_time_step, self.simulation_duration,
                                        self.simulation_time_step))),
            self.simulation_duration)

        if len(simulation_time) > len(simulation_e_field_magnitude):
            simulation_e_field_magnitude = np.pad(simulation_e_field_magnitude,
                                                  (0, len(simulation_time) - len(simulation_e_field_magnitude)),
                                                  constant_values=(0, 0))
        else:
            simulation_e_field_magnitude = simulation_e_field_magnitude[:len(simulation_time)]

        return simulation_e_field_magnitude, simulation_time


    def _load_waveform(self, waveform_type: WaveformType):

        tstop = self.simulation_duration
        delay = self.stimulation_delay
        dt = self.simulation_time_step

        kind = WAVE_TYPE_STR[waveform_type]

        tms_waves = loadmat(
            str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath('coil_recordings/TMSwaves.mat').absolute())
        )

        recorded_time = tms_waves[f't{kind}'].ravel() + delay
        recorded_e_field_magnitude = tms_waves[f'Erec_{kind}'].flatten()

        interp = interp1d(recorded_time, recorded_e_field_magnitude,
                          bounds_error=False, fill_value=0.0)
        
        t = np.arange(0, tstop+dt, dt)

        return interp(t), t
