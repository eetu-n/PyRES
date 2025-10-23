# ==================================================================
# ============================ IMPORTS =============================
from collections import OrderedDict
# PyTorch
import torch
import torch.nn as nn
# FLAMO
from flamo import dsp
from flamo.functional import mag2db, WGN_reverb
# PyRES
from PyRES.dataset_api import (
    get_hl_info,
    get_ll_info,
    get_rirs,
    normalize_rirs,
    get_transducer_number,
    get_transducer_positions
)
from PyRES.functional import energy_coupling, direct_to_reverb_ratio
from PyRES.functional import simulate_setup
from PyRES.plots import (
    plot_room_setup,
    plot_coupling,
    plot_DRR,
    plot_distributions
)


# ==================================================================
# =========================== BASE CLASS ===========================

class PhRoom(object):
    r"""
    Base class for physical-room implementations.
    """
    def __init__(
            self,
            fs: int,
            nfft: int,
            alias_decay_db: float,
            device: torch.device = torch.get_default_device()
        ) -> None:
        r"""
        Initializes the PhRoom object.

            **Args**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - the device to learn with, usually "cuda" or "cpu"

            **Attributes**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - transducer_number (OrderedDict): Number of transducers in the room.
                - transducer_indices (OrderedDict): Indices of the requested stage emitters, system receivers, system emitters and audience receivers.
                - transducer_positions (OrderedDict): Positions of the requested stage emitters, system receivers, system emitters and audience receivers.
                - rir_length (int): Length of the room impulse responses in samples.
                - h_SA (nn.Module): Room impulse responses bewteen stage emitters and audience receivers.
                - h_SM (nn.Module): Room impulse responses bewteen stage emitters and system receivers.
                - h_LA (nn.Module): Room impulse responses bewteen system emitters and audience receivers.
                - h_LM (nn.Module): Room impulse responses bewteen system emitters and system receivers.
        """
        object.__init__(self)

        self.fs = fs
        self.nfft = nfft
        self.alias_decay_db = alias_decay_db

        self.device = device

        self.transducer_number = OrderedDict(
            {'stg': int, 'mcs': int, 'lds': int, 'aud': int}
        )

        self.transducer_indices = OrderedDict(
            {'stg': list[int], 'mcs': list[int], 'lds': list[int], 'aud': list[int]}
        )

        self.transducer_positions = OrderedDict(
            {'stg': list[list[int]], 'mcs': list[list[int]], 'lds': list[list[int]], 'aud': list[list[int]]}
        )

        self.rir_length: int

        self.h_SA: nn.Module
        self.h_SM: nn.Module
        self.h_LA: nn.Module
        self.h_LM: nn.Module

        self.energy_coupling = OrderedDict(
            {'SA': torch.Tensor, 'SM': torch.Tensor, 'LM': torch.Tensor, 'LA': torch.Tensor}
        )

        self.direct_to_reverb_ratio = OrderedDict(
            {'SA': torch.Tensor, 'SM': torch.Tensor, 'LM': torch.Tensor, 'LA': torch.Tensor}
        )

    def get_ems_rcs_number(self) -> OrderedDict:
        r"""
        Returns the number of emitters and receivers.

            **Returns**:
                - OrderedDict: Number of emitters and receivers.
        """
        return self.transducer_number
    
    def get_stg_to_aud(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between stage emitters and audience receivers.

            **Returns**:
                - torch.Tensor: Stage-to-Audience RIRs. shape = (samples, n_A, n_S).
        """
        return self.h_SA

    def get_stg_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between stage emitters and system receivers.

            **Returns**:
                - torch.Tensor: Stage-to-Microphones RIRs. shape = (samples, n_M, n_S).
        """
        return self.h_SM
    
    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between system emitters and audience receivers.

            **Returns**:
                - torch.Tensor: Loudspeakers-to-Audience RIRs. shape = (samples n_A, n_L).
        """
        return self.h_LA

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between system emitters and system receivers.

            **Returns**:
                - torch.Tensor: Loudspeakers-to-Microphones RIRs. shape = (samples, n_M, n_L).
        """
        return self.h_LM
    
    def get_rirs(self) -> OrderedDict:
        r"""
        Returns a copy of all system room impulse responses.

            **Returns**:
                - OrderedDict: System RIRs.
        """
        RIRs = OrderedDict()
        RIRs.update({'SM': self.get_stg_to_mcs().param.clone().detach()})
        RIRs.update({'SA': self.get_stg_to_aud().param.clone().detach()})
        RIRs.update({'LM': self.get_lds_to_mcs().param.clone().detach()})
        RIRs.update({'LA': self.get_lds_to_aud().param.clone().detach()})
        return RIRs
    
    def compute_energy_coupling(self) -> OrderedDict:
        r"""
        Computes the energy coupling of the room impulse responses.

            **Returns**:
                - OrderedDict: Energy coupling of the system RIRs.
        """
        rirs = self.get_rirs()
        ec_SA = energy_coupling(rirs["SA"], fs=self.fs)
        ec_SM = energy_coupling(rirs["SM"], fs=self.fs)
        ec_LM = energy_coupling(rirs["LM"], fs=self.fs)
        ec_LA = energy_coupling(rirs["LA"], fs=self.fs)

        ec = OrderedDict()
        ec.update({'SA': ec_SA})
        ec.update({'SM': ec_SM})
        ec.update({'LM': ec_LM})
        ec.update({'LA': ec_LA})

        return ec
    
    def compute_direct_to_reverb_ratio(self) -> OrderedDict:
        r"""
        Computes the direct-to-reverberant ratio (DRR) of the room impulse responses.

            **Returns**:
                - OrderedDict: Direct-to-reverberant ratio of the system RIRs.
        """
        rirs = self.get_rirs()
        drr_SA = direct_to_reverb_ratio(rirs["SA"], fs=self.fs)
        drr_SM = direct_to_reverb_ratio(rirs["SM"], fs=self.fs)
        drr_LM = direct_to_reverb_ratio(rirs["LM"], fs=self.fs)
        drr_LA = direct_to_reverb_ratio(rirs["LA"], fs=self.fs)

        drr = OrderedDict()
        drr.update({'SA': drr_SA})
        drr.update({'SM': drr_SM})
        drr.update({'LM': drr_LM})
        drr.update({'LA': drr_LA})

        return drr
    
    def create_modules(self,
            rirs_SA: torch.Tensor,
            rirs_SM: torch.Tensor,
            rirs_LA: torch.Tensor,
            rirs_LM: torch.Tensor,
            rir_length: int
        ) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter]:
        r"""
        Creates the processing modules for the room-impulse-response blocks.

            **Args**:
                - rirs_SA (torch.Tensor): Room impulse responses between stage emitters and audience receivers.
                - rirs_SM (torch.Tensor): Room impulse responses between stage emitters and system receivers.
                - rirs_LA (torch.Tensor): Room impulse responses between system emitters and audience receivers.
                - rirs_LM (torch.Tensor): Room impulse responses between system emitters and system receivers.
                - rir_length (int): Length of the room impulse responses in samples.

            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen stage emitters and system receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and system receivers.
        """
        # Get number of transducers
        n_S = self.transducer_number['stg']
        n_M = self.transducer_number['mcs']
        n_L = self.transducer_number['lds']
        n_A = self.transducer_number['aud']

        # Stage to Audience
        h_SA = dsp.Filter(
            size=(rir_length, n_A, n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device = self.device
        )
        h_SA.assign_value(rirs_SA)
        h_SA.to(self.device)

        # Stage to Microphones
        h_SM = dsp.Filter(
            size=(rir_length, n_M, n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device = self.device
        )
        h_SM.assign_value(rirs_SM)
        h_SM.to(self.device)

        # Loudspeakers to Audience
        h_LM = dsp.Filter(
            size=(rir_length, n_M, n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device = self.device
        )
        h_LM.assign_value(rirs_LM)
        h_LM.to(self.device)

        # Loudspeakers to Microphones
        h_LA = dsp.Filter(
            size=(rir_length, n_A, n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
            device = self.device
        )
        h_LA.assign_value(rirs_LA)
        h_LA.to(self.device)

        return h_SA, h_SM, h_LA, h_LM
    
    def plot_setup(self) -> None:
        r"""
        Plots the room setup.
        """
        plot_room_setup(self.transducer_positions)
        return None
    
    def plot_coupling(self) -> None:
        r"""
        Plots the room coupling.
        """
        plot_coupling(energy_values=self.energy_coupling)
        return None
    
    def plot_DRR(self) -> None:
        r"""
        Plots the direct-to-reverberant ratio (DRR).
        """
        plot_DRR(direct_to_reverb_ratios=self.direct_to_reverb_ratio)
        return None

    def plot_h_LM_distributions(self, db_scale: bool=False) -> None:
        r"""
        Plots the distributions of the room impulse responses between system emitters and system receivers.

            **Args**:
                - db_scale (bool): If True, the imaginary part is converted to dB scale.
        """
        h_LM = self.get_rirs()['LM']
        H_LM = torch.fft.rfft(h_LM, n=self.nfft, dim=0)
        real = torch.real(H_LM[1:-1, :, :])  # Exclude the DC component and Nyquist frequency
        imag = torch.imag(H_LM[1:-1, :, :])
        if db_scale:
            real = mag2db(real + 1e-10)
            imag = mag2db(imag + 1e-10)

        distributions = torch.stack((real.flatten(), imag.flatten()), dim=1)

        plot_distributions(distributions=distributions, n_bins=self.nfft//100, labels=['Real', 'Imaginary'])

        return None


# ==================================================================
# ========================== DATASET CLASS =========================

class PhRoom_dataset(PhRoom):
    r"""
    Subclass of PhRoom that loads the room impulse responses from the dataset.
    """
    def __init__(
            self,
            fs: int,
            nfft: int,
            alias_decay_db: float,
            dataset_directory: str,
            room_name: str,
            device: torch.device = torch.get_default_device(),
            stg_idx: list[int] = None,
            mcs_idx: list[int] = None,
            lds_idx: list[int] = None,
            aud_idx: list[int] = None
        ) -> None:
        r"""
        Initializes the PhRoom_dataset object.

            **Args**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - dataset_directory (str): Path to the dataset.
                - room_name (str): Name of the room.
                - stg_idx (list[int]): List of indices of the requested stage emitters.
                - mcs_idx (list[int]): List of indices of the requested system receivers.
                - lds_idx (list[int]): List of indices of the requested system emitters.
                - aud_idx (list[int]): List of indices of the requested audience receivers.

            **Attributes**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - room_name (str): Name of the room.
                - high_level_info (dict): High-level information of the room.
                - low_level_info (dict): Low-level information of the room.
                - transducer_number (OrderedDict): Number of transducers in the room.
                - transducer_indices (OrderedDict): Indices of the requested stage emitters, system receivers, system emitters and audience receivers.
                - transducer_positions (OrderedDict): Positions of the requested stage emitters, system receivers, system emitters and audience receivers.
                - rir_length (int): Length of the room impulse responses in samples.
                - h_SA (nn.Module): Room impulse responses bewteen stage emitters and audience receivers.
                - h_SM (nn.Module): Room impulse responses bewteen stage emitters and system receivers.
                - h_LA (nn.Module): Room impulse responses bewteen system emitters and audience receivers.
                - h_LM (nn.Module): Room impulse responses bewteen system emitters and system receivers.
        """
        super().__init__(
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db,
            device = device
        )

        self.room_name = room_name

        self.device = device

        self.high_level_info = get_hl_info(
            ds_dir=dataset_directory,
            room=self.room_name
        )

        self.room_directory = self.high_level_info['RoomDirectory']

        self.low_level_info = get_ll_info(
            ds_dir=dataset_directory,
            room_dir=self.room_directory
        )

        self.transducer_number, self.transducer_indices = get_transducer_number(
            ll_info=self.low_level_info,
            stg_idx=stg_idx,
            mcs_idx=mcs_idx,
            lds_idx=lds_idx,
            aud_idx=aud_idx
        )

        self.transducer_positions = get_transducer_positions(
            ll_info=self.low_level_info,
            stg_idx=self.transducer_indices['stg'],
            mcs_idx=self.transducer_indices['mcs'],
            lds_idx=self.transducer_indices['lds'],
            aud_idx=self.transducer_indices['aud']
        )

        self.h_SA, self.h_SM, self.h_LA, self.h_LM, self.rir_length = self.__load_rirs(
            ds_dir=dataset_directory
        )

        self.energy_coupling = self.compute_energy_coupling()
        self.direct_to_reverb_ratio = self.compute_direct_to_reverb_ratio()

    def __load_rirs(self, ds_dir: str) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter, int]:
        r"""
        Loads all the room impulse responses from the dataset and returns them in processing modules.

            **Args**:
                - ds_dir (str): Path to the dataset.
            
            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen stage emitters and system receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and system receivers.
                - int: Length of the room impulse responses in samples.
        """
        # Load RIRs
        rirs, rir_length = get_rirs(
            ds_dir=ds_dir,
            room_dir=self.room_directory,
            transducer_indices=self.transducer_indices,
            target_fs=self.fs,
            device = self.device
        )

        # Energy normalization
        rirs_norm = normalize_rirs(
            fs=self.fs,
            stg_to_aud=rirs["stg_to_aud"],
            stg_to_sys=rirs["stg_to_sys"],
            sys_to_aud=rirs["sys_to_aud"],
            sys_to_sys=rirs["sys_to_sys"]
        )

        # Create processing modules
        h_SA, h_SM, h_LA, h_LM = self.create_modules(
            rirs_SA=rirs_norm["stg_to_aud"],
            rirs_SM=rirs_norm["stg_to_sys"],
            rirs_LA=rirs_norm["sys_to_aud"],
            rirs_LM=rirs_norm["sys_to_sys"],
            rir_length=rir_length
        )

        return h_SA, h_SM, h_LA, h_LM, rir_length
    
# ==================================================================
# =================== WHITE GAUSSIAN NOISE CLASS ===================

class PhRoom_wgn(PhRoom):
    r"""
    Subclass of PhRoom that synthetize the room impulse responses of a RES setup in a shoebox room.
    The room is defined by its size and reverberation time.
    The setup includes one stage emitter and one audience receiver only.
    The room impulse responses are approximated to late reverberation only as exponentially-decaying white-Gaussian-noise sequences.
    """
    def __init__(
            self,
            fs: int,
            nfft: int,
            alias_decay_db: float,
            room_dims: tuple[float, float, float],
            room_RT: float,
            n_M: int,
            n_L: int,
            method: str = 'Poletti'
        ) -> None:
        r"""
        Initializes the PhRoom_wgn object.

            **Args**:
                - room_size (tuple[float, float, float]): Room size in meters.
                - room_RT (float): Room reverberation time [s].
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - n_L (int): Number of system loudspeakers.
                - n_M (int): Number of system microphones.
        """
        assert n_M > 0,  "The number of system microphones must be higher than 0."
        assert n_L > 0,  "The number of system loudspeakers must be higher than 0."

        super().__init__(
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.room_dims = torch.FloatTensor(room_dims)
        self.RT = room_RT

        self.transducer_number = OrderedDict()
        self.transducer_number.update({'stg': 1})
        self.transducer_number.update({'mcs': n_M})
        self.transducer_number.update({'lds': n_L})
        self.transducer_number.update({'aud': 1})
        
        self.transducer_positions = simulate_setup(
            room_dims=self.room_dims,
            mcs_n=n_M,
            lds_n=n_L
        )

        if method not in ['Poletti', 'Barron']:
            raise ValueError(f"Method '{method}' is not supported. Choose 'Poletti' or 'Barron'.")
        self.method = method

        self.h_SA, self.h_SM, self.h_LA, self.h_LM, self.rir_length = self.__generate_rirs()

        self.energy_coupling = self.compute_energy_coupling()
        self.direct_to_reverb_ratio = self.compute_direct_to_reverb_ratio()

    def __generate_rirs(self) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter, int]:
        r"""
        Generates the room impulse responses of the RES setup.

            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen stage emitters and system receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and system receivers.
                - int: Length of the room impulse responses in samples.
        """
        
        rirs_SA = self.__generate_rirs_of(
            n_emitters=self.transducer_number['stg'],
            pos_emitters=self.transducer_positions['stg'],
            n_receivers=self.transducer_number['aud'],
            pos_receivers=self.transducer_positions['aud']
        )
        rirs_SM = self.__generate_rirs_of(
            n_emitters=self.transducer_number['stg'],
            pos_emitters=self.transducer_positions['stg'],
            n_receivers=self.transducer_number['mcs'],
            pos_receivers=self.transducer_positions['mcs']
        )
        rirs_LA = self.__generate_rirs_of(
            n_emitters=self.transducer_number['lds'],
            pos_emitters=self.transducer_positions['lds'],
            n_receivers=self.transducer_number['aud'],
            pos_receivers=self.transducer_positions['aud']
        )
        rirs_LM = self.__generate_rirs_of(
            n_emitters=self.transducer_number['lds'],
            pos_emitters=self.transducer_positions['lds'],
            n_receivers=self.transducer_number['mcs'],
            pos_receivers=self.transducer_positions['mcs']
        )

        # Get the length of the RIRs
        sa_length = rirs_SA.shape[0]
        sm_length = rirs_SM.shape[0]
        lm_length = rirs_LM.shape[0]
        la_length = rirs_LA.shape[0]

        max_length = max(sa_length, sm_length, lm_length, la_length)

        # Pad the RIRs to the same length
        rirs_SA = torch.nn.functional.pad(
            input=rirs_SA,
            pad=(0, 0, 0, 0, 0, max_length - sa_length),
            mode='constant',
            value=0
        )
        rirs_SM = torch.nn.functional.pad(
            input=rirs_SM,
            pad=(0, 0, 0, 0, 0, max_length - sm_length),
            mode='constant',
            value=0
        )
        rirs_LA = torch.nn.functional.pad(
            input=rirs_LA,
            pad=(0, 0, 0, 0, 0, max_length - la_length),
            mode='constant',
            value=0
        )
        rirs_LM = torch.nn.functional.pad(
            input=rirs_LM,
            pad=(0, 0, 0, 0, 0, max_length - lm_length),
            mode='constant',
            value=0
        )

        # Create processing modules
        h_SA, h_SM, h_LA, h_LM = self.create_modules(
            rirs_SA=rirs_SA,
            rirs_SM=rirs_SM,
            rirs_LA=rirs_LA,
            rirs_LM=rirs_LM,
            rir_length=max_length
        )

        return h_SA, h_SM, h_LA, h_LM, max_length


    def __generate_rirs_of(self,
            n_emitters: int,
            pos_emitters: torch.FloatTensor,
            n_receivers: int,
            pos_receivers: torch.FloatTensor
        ) -> torch.Tensor:
        r"""
        Generates the room impulse responses between the emitters and receivers.

            **Args**:
                - n_emitters (int): Number of emitters.
                - pos_emitters (torch.FloatTensor): Positions of the emitters.
                - n_receivers (int): Number of receivers.
                - pos_receivers (torch.FloatTensor): Positions of the receivers.

            **Returns**:
                - torch.Tensor: Room impulse responses.
        """
        # Generate the room impulse responses as exponentially-decaying white-Gaussian-noise sequences
        rirs = WGN_reverb(
            matrix_size=(n_receivers, n_emitters),
            t60=self.RT,
            samplerate=self.fs,
        )

        # Compute the distances between the emitters and receivers
        pos_emitters = pos_emitters.unsqueeze(0).repeat(n_receivers,1,1)
        pos_receivers = pos_receivers.unsqueeze(1).repeat(1,n_emitters,1)
        distances = torch.linalg.norm(pos_emitters - pos_receivers, dim=2)

        # Compute the propagation times and convert them to samples
        speed_of_sound = 343  # m/s
        propagation_times = distances / speed_of_sound
        propagation_samples = torch.round(propagation_times * self.fs).long()

        # Zero-pad the RIRs
        rirs = torch.nn.functional.pad(
            input=rirs,
            pad=(0, 0, 0, 0, 0, propagation_samples.max().item()),
            mode='constant',
            value=0
        )

        # Shift the RIRs according to the propagation times
        for r in range(n_receivers):
            for e in range(n_emitters):
                rirs[:,r,e] = torch.roll(rirs[:,r,e], shifts=propagation_samples[r,e].item(), dims=0)

        return rirs
    
    def regenerate_h_LM(self) -> None:
        r"""
        Regenerates the room impulse responses between system emitters and system receivers.
        """
        new_rirs = self.__generate_rirs_of(
                n_emitters=self.transducer_number['lds'],
                pos_emitters=self.transducer_positions['lds'],
                n_receivers=self.transducer_number['mcs'],
                pos_receivers=self.transducer_positions['mcs']
            )
        self.h_LM.assign_value(new_rirs)
        self.h_LM.to(self.device)

        self.energy_coupling['LM'] = energy_coupling(new_rirs, fs=self.fs)
        self.direct_to_reverb_ratio['LM'] = direct_to_reverb_ratio(new_rirs, fs=self.fs)

        return None