import torch
import torch.nn as nn
from flamo import dsp, system
from flamo.functional import WGN_reverb
from flamo.utils import to_complex
from functional import modal_reverb, FDN_absorption

class unitary_connections(system.Series):
    def __init__(
        self,
        n_M: int = 1,
        n_L: int = 1,
        nfft: int = 2**11,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
    ):
        
        max_dimension = torch.max(torch.tensor([n_M, n_L]))

        I1 = dsp.Matrix(
            size = (max_dimension, n_M),
            nfft = nfft,
            matrix_type = "identity",
            requires_grad = False,
            alias_decay_db = alias_decay_db,
        )
        I1.assign_value(torch.eye(max_dimension, n_M))

        U = dsp.Matrix(
            size = (max_dimension, max_dimension),
            nfft = nfft,
            matrix_type = "orthogonal",
            requires_grad = requires_grad,
            alias_decay_db = alias_decay_db,
        )

        I2 = dsp.Matrix(
            size = (n_L, max_dimension),
            nfft = nfft,
            matrix_type = "identity",
            requires_grad = False,
            alias_decay_db = alias_decay_db,
        )
        I2.assign_value(torch.eye(n_L, max_dimension))

        super().__init__(I1, U, I2)


class random_FIRs(dsp.Filter):
    """
    Reference:
        De Bortoli G., Dal Santo G., Prawda K., Lokki T., Välimäki V., and Schlecht S. J.
        "Differentiable Active Acoustics---Optimizing Stability via Gradient Descent"
        International Conference on Digital Audio Effects (DAFx), Sep. 2024, Guilford, UK.
    """
    def __init__(
            self,
            n_M: int=1,
            n_L: int=1,
            nfft: int=2**11,
            FIR_order: int=100,
            alias_decay_db: float=0.0,
            requires_grad: bool=False
        ) -> None:
        
        super().__init__(size=(FIR_order, n_L, n_M), nfft=nfft, requires_grad=requires_grad, alias_decay_db=alias_decay_db)


class phase_canceling_modal_reverb(dsp.DSP):
    """
    Reference:
        De Bortoli G., Prawda K., and Schlecht S. J.
        "Active Acoustics With a Phase Cancelling Modal Reverberator"
        Journal of the Audio Engineering Society 72, no. 10 (2024): 705-715.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int = 48000,
        nfft: int = 2**11,
        n_modes: int=10,
        low_f_lim: float=0,
        high_f_lim: float=500,
        t60: float=1.0,
        requires_grad: bool=False,
        alias_decay_db: float=0.0,
    ):
        super().__init__(size=(1, n_L, n_M, n_modes,), nfft=nfft, requires_grad=requires_grad, alias_decay_db=alias_decay_db)

        self.fs = fs
        self.n_modes = n_modes
        self.resonances = torch.linspace(low_f_lim, high_f_lim, n_modes).view(
            -1, *(1,)*(len(self.param.shape[:-1]))).permute(
                [1,2,3,0]).expand(
                    *self.param.shape)
        self.gains = torch.ones_like(self.param)
        self.t60 = t60 * torch.ones_like(self.param)
        self.initialize_class()

    def forward(self, x, ext_param=None):
        r"""
        Applies the Filter module to the input tensor x.

            **Arguments**:
                - **x** (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
                - **ext_param** (torch.Tensor, optional): Parameter values received from external modules (hyper conditioning). Default: None.

            **Returns**:
                torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        self.check_input_shape(x)
        if ext_param is None:
            return self.freq_convolve(x, self.param)
        else:
            # log the parameters that are being passed
            with torch.no_grad():
                self.assign_value(ext_param)
            return self.freq_convolve(x, ext_param)
        
    def check_input_shape(self, x):
        r"""
        Checks if the dimensions of the input tensor x are compatible with the module.

            **Arguments**:
                **x** (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
        """
        if (int(self.nfft / 2 + 1), self.input_channels) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.freq_response.shape} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        r"""
        Checks if the shape of the filter parameters is valid.
        """
        assert (
            len(self.size) == 4
        ), "Filter must be 3D, for 2D (parallel) filters use ParallelFilter module."

    
    def init_param(self):
        torch.nn.init.uniform_(self.param, a=0, b=2*torch.pi)

    def get_freq_response(self):

        self.freq_response = lambda param: modal_reverb(fs=self.fs, nfft=self.nfft, resonances=self.resonances, gains=self.gains, phases=param, t60=self.t60, alias_decay_db=self.alias_decay_db)

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.

        The frequency convolution is computed using the :func:`torch.einsum` function.

            **Arguments**:
                **x** (torch.Tensor): Input tensor.

            **Returns**:
                torch.Tensor: Output tensor after frequency convolution.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response(param), x
        )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-2]
        self.output_channels = self.size[-3]

    def initialize_class(self):
        self.init_param()
        self.get_gamma()
        self.check_param_shape()
        self.get_io()
        self.get_freq_response()
        self.get_freq_convolve()

class feedback_delay_network(system.Series):
        
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int=48000,
        nfft: int=2**11,
        t60_DC: float=1.0,
        t60_NY: float=1.0,
        alias_decay_db: float=0.0,
        requires_grad: bool=False
    ):

        max_dimension = torch.max(torch.tensor([n_M, n_L]))

        input_gains = dsp.Gain(
            size = (max_dimension, n_M),
            nfft = nfft,
            matrix_type = "identity",
            requires_grad = False,
            alias_decay_db = alias_decay_db,
        )
        input_gains.assign_value(torch.eye(max_dimension, n_M))

        delays = dsp.parallelDelay(
            size = (max_dimension,),
            max_len = 2000,
            nfft = nfft,
            isint = True,
            requires_grad = False,
            alias_decay_db = alias_decay_db,
        )

        feedback_matrix = dsp.Matrix(
            size = (max_dimension, max_dimension),
            nfft = nfft,
            matrix_type = "orthogonal",
            requires_grad = requires_grad,
            alias_decay_db = alias_decay_db,
        )

        attenuation = FDN_absorption(
            delays = delays.param,
            fs = fs,
            nfft = nfft,
            t60_DC = t60_DC,
            t60_NY = t60_NY,
            alias_decay_db = alias_decay_db,
        )

        recursion = system.Recursion(
            fF = system.Series(delays, attenuation),
            fB = feedback_matrix,
        )

        output_gains = dsp.Gain(
            size = (n_L, max_dimension),
            nfft = nfft,
            matrix_type = "identity",
            requires_grad = False,
            alias_decay_db = alias_decay_db,
        )
        output_gains.assign_value(torch.eye(n_L, max_dimension))

        super().__init__(output_gains, input_gains, recursion, output_gains, input_gains)


class unitary_allpass_reverb():
    pass


class wgn_reverb(dsp.DSP):
    # TODO: After modifying wgn_reverb in FLAMO, this class has RT and mean and/or std as parameters to be optimized
    # TODO: wgn_reverb is used in get_freq_response()
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        nfft: int=2**11,
        RT: float=1.0,
        mean: float=0.0,
        std: float=1.0,
        requires_grad: bool=False
    ):
        pass
    pass