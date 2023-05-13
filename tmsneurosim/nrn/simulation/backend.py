class __Backend:
    """
    NEURON backend. Set global tstop, dt, rhoe, and temp
    parameters.

    DEFAULTS
    --------
    dt        = 0.005 [ms]

    tstop     = 1     [ms]

    temp      = 37    [C]

    delay     = 0.005 [ms]

    threshold = 0     [mV]

    validate  = False [bool]

    ap_dur    = 0.3   [ms]

    """
    __defaults__ = {
        'dt': 0.005,
        'tstop': 1.0,
        'temp': 37,
        'delay': 0.005,
        'threshold': 0.0,
        'validate': False,
        'ap_dur': 0.3
    }

    _instance = None  # Keep instance reference

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):

        for k, v in self.__defaults__.items():
            setattr(self, f'_{k}', v)

    @property
    def dt(self):
        """Global simulation timestep."""
        return self._dt

    @dt.setter
    def dt(self, value):
        """Set dt globally."""
        self._dt = value

    @property
    def threshold(self):
        """Global simulation threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """Set threshold globally."""
        self._threshold = value

    @property
    def tstop(self):
        """Global simulation duration."""
        return self._tstop

    @tstop.setter
    def tstop(self, value):
        """Set simulation duration globally."""
        self._tstop = value

    @property
    def temp(self):
        """Global simulation timestep."""
        return self._temp

    @temp.setter
    def temp(self, value):
        """Set temp globally."""
        self._temp = value

    @property
    def delay(self):
        """Global simulation delay."""
        return self._delay

    @delay.setter
    def delay(self, value):
        """Set delay globally."""
        self._delay = value

    @property
    def validate(self):
        return self._validate
    
    @validate.setter
    def validate(self, value):
        self._validate = value

    @property
    def ap_dur(self):
        return self._ap_dur
    
    @ap_dur.setter
    def ap_dur(self, value):
        self._ap_dur = value

    def reset(self, quantity=None):
        if quantity is None:
            for k, v in self.__defaults__.items():
                setattr(self, k, v)
        else:
            try:
                setattr(self, quantity, self.__defaults__[quantity])
            except KeyError:
                raise ValueError(f'nrn Backend has no default for {quantity}') from None


Backend = __Backend()