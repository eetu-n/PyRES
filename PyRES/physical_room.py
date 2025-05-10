# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
from collections import OrderedDict
import json
# PyTorch
import torch
import torch.nn as nn
import torchaudio
# FLAMO
from flamo import dsp
# PyRES
from PyRES.metrics import energy_coupling
from PyRES.plots import plot_room_setup, plot_coupling, plot_DRR


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
            alias_decay_db: float
        ) -> None:
        r"""
        Initializes the PhRoom object.

            **Args**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].

            **Attributes**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - n_S (int): Number of stage sources.
                - n_L (int): Number of system loudspeakers.
                - n_M (int): Number of system microphones.
                - n_A (int): Number of audience positions.
                - rir_length (int): Length of the room impulse responses in samples.
                - h_SA (nn.Module): Room impulse responses bewteen stage sources and audience positions.
                - h_SM (nn.Module): Room impulse responses bewteen stage sources and system microphones.
                - h_LA (nn.Module): Room impulse responses bewteen system loudspeakers and audience positions.
                - h_LM (nn.Module): Room impulse responses bewteen system loudspeakers and system microphones.
        """
        object.__init__(self)

        self.fs = fs
        self.nfft = nfft
        self.alias_decay_db = alias_decay_db

        self.n_S: int
        self.n_L: int
        self.n_M: int
        self.n_A: int

        self.idx_S: list[int]
        self.idx_L: list[int]
        self.idx_M: list[int]
        self.idx_A: list[int]

        self.pos_S: torch.Tensor
        self.pos_L: torch.Tensor
        self.pos_M: torch.Tensor
        self.pos_A: torch.Tensor

        self.rir_length: int
        self.h_SA: nn.Module
        self.h_SM: nn.Module
        self.h_LA: nn.Module
        self.h_LM: nn.Module

    def get_ems_rcs_number(self) -> OrderedDict:
        r"""
        Returns the number of emitters and receivers.

            **Returns**:
                - OrderedDict: Number of emitters and receivers.
        """
        return self.n_S, self.n_M, self.n_L, self.n_A
    
    def get_stg_to_aud(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between stage sources and audience positions.

            **Returns**:
                - torch.Tensor: Stage-to-Audience RIRs. shape = (samples, n_A, n_S).
        """
        return self.h_SA

    def get_stg_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between stage sources and system microphones.

            **Returns**:
                - torch.Tensor: Stage-to-Microphones RIRs. shape = (samples, n_M, n_S).
        """
        return self.h_SM
    
    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between system loudspeakers and audience positions.

            **Returns**:
                - torch.Tensor: Loudspeakers-to-Audience RIRs. shape = (samples n_A, n_L).
        """
        return self.h_LA

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between system loudspeakers and system microphones.

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
        RIRs.update({'h_SM': self.get_stg_to_mcs().param.clone().detach()})
        RIRs.update({'h_SA': self.get_stg_to_aud().param.clone().detach()})
        RIRs.update({'h_LM': self.get_lds_to_mcs().param.clone().detach()})
        RIRs.update({'h_LA': self.get_lds_to_aud().param.clone().detach()})
        return RIRs
    
    def create_modules(self, rirs_SA: torch.Tensor, rirs_SM: torch.Tensor, rirs_LA: torch.Tensor, rirs_LM: torch.Tensor, rir_length: int) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter]:
        r"""
        Creates the processing modules for the room-impulse-response blocks.

            **Args**:
                - rirs_SA (torch.Tensor): Room impulse responses between stage sources and audience positions.
                - rirs_SM (torch.Tensor): Room impulse responses between stage sources and system microphones.
                - rirs_LA (torch.Tensor): Room impulse responses between system loudspeakers and audience positions.
                - rirs_LM (torch.Tensor): Room impulse responses between system loudspeakers and system microphones.
                - rir_length (int): Length of the room impulse responses in samples.

            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage sources and audience positions.
                - dsp.Filter: Room impulse responses bewteen stage sources and system microphones.
                - dsp.Filter: Room impulse responses bewteen system loudspeakers and audience positions.
                - dsp.Filter: Room impulse responses bewteen system loudspeakers and system microphones.
        """

        # Stage to Audience
        h_SA = dsp.Filter(
            size=(rir_length, self.n_A, self.n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_SA.assign_value(rirs_SA)

        # Stage to Microphones
        h_SM = dsp.Filter(
            size=(rir_length, self.n_M, self.n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_SM.assign_value(rirs_SM)

        # Loudspeakers to Audience
        h_LM = dsp.Filter(
            size=(rir_length, self.n_M, self.n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_LM.assign_value(rirs_LM)

        # Loudspeakers to Microphones
        h_LA = dsp.Filter(
            size=(rir_length, self.n_A, self.n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_LA.assign_value(rirs_LA)

        return h_SA, h_SM, h_LA, h_LM
    
    def plot_setup(self) -> None:
        r"""
        Plots the room setup.
        """
        plot_room_setup(self.pos_S, self.pos_M, self.pos_L, self.pos_A)
    
    def plot_coupling(self) -> None:
        r"""
        Plots the room coupling.
        """
        plot_coupling(rirs=self.get_rirs(), fs=self.fs)
    
    def plot_DRR(self) -> None:
        r"""
        Plots the direct-to-reverberant ratio (DRR).
        """
        plot_DRR(rirs=self.get_rirs(), fs=self.fs)


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
            lds_idx: list[int] = None,
            mcs_idx: list[int] = None,
        ) -> None:
        r"""
        Initializes the PhRoom_dataset object.

            **Args**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - dataset_directory (str): Path to the dataset.
                - room_name (str): Name of the room.
                - lds_idx (list[int]): List of indices of the requested system loudspeakers.
                - mcs_idx (list[int]): List of indices of the requested system microphones.

            **Attributes**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - room_name (str): Name of the room.
                - high_level_info (dict): High-level information of the room.
                - low_level_info (dict): Low-level information of the room.
                - n_S (int): Number of stage sources.
                - n_L (int): Number of system loudspeakers.
                - n_M (int): Number of system microphones.
                - n_A (int): Number of audience positions.
                - rir_length (int): Length of the room impulse responses in samples.
                - h_SA (nn.Module): Room impulse responses bewteen stage sources and audience positions.
                - h_SM (nn.Module): Room impulse responses bewteen stage sources and system microphones.
                - h_LA (nn.Module): Room impulse responses bewteen system loudspeakers and audience positions.
                - h_LM (nn.Module): Room impulse responses bewteen system loudspeakers and system microphones.
        """
        super().__init__(
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.room_name = room_name
        self.high_level_info = self.__find_room_in_dataset(
            ds_dir=dataset_directory,
            room=room_name
        )

        self.low_level_info = self.__get_room_info(
            ds_dir=dataset_directory,
            room_dir=self.high_level_info['RoomDirectory']
        )

        self.n_S, self.n_M, self.n_L, self.n_A, self.idx_S, self.idx_M, self.idx_L, self.idx_A = self.__ems_rcs_number(lds_idx=lds_idx, mcs_idx=mcs_idx)
        self.pos_S, self.pos_M, self.pos_L, self.pos_A = self.__ems_rcs_positions()
        self.h_SA, self.h_SM, self.h_LA, self.h_LM, self.rir_length = self.__load_rirs(
            ds_dir=dataset_directory
        )

    def __find_room_in_dataset(self, ds_dir: str, room: str) -> dict:
        r"""
        Finds the room in the dataset.

            **Args**:
                - ds_dir (str): Path to the dataset.
                - room (str): Name of the room.

            **Returns**:
                - dict: High-level information of the room.
        """
        ds_dir = ds_dir.rstrip('/')
        with open(f"{ds_dir}/datasetInfo.json", 'r') as file:
            data = json.load(file)
        
        return data['Rooms'][room]
    
    def __get_room_info(self, ds_dir: str, room_dir: str) -> dict:
        r"""
        Gets the room information.

            **Args**:
                - ds_dir (str): Path to the dataset.
                - room_dir (str): Path to the room in the dataset.

            **Returns**:
                - dict: Low-level information of the room.
        """
        ds_dir = ds_dir.rstrip('/')
        with open(f"{ds_dir}/data/{room_dir}/roomInfo.json", 'r') as file:
            data = json.load(file)

        return data
    
    def __ems_rcs_number(self, lds_idx: list[int]=None, mcs_idx: list[int]=None) -> tuple[int, int, int, int]:
        r"""
        Scans the room information for the number of emitters and receivers.

            **Returns**:
                - int: Number of stage sources.
                - int: Number of system microphones.
                - int: Number of system loudspeakers.
                - int: Number of audience positions.
        """
        n_S = self.low_level_info['StageAndAudience']['StageEmitters']['Number']
        n_M = self.low_level_info['AudioSetup']['SystemReceivers']['Number']
        n_L = self.low_level_info['AudioSetup']['SystemEmitters']['Number']
        n_A = self.low_level_info['StageAndAudience']['AudienceReceivers']['MonochannelNumber']
        idx_S = [i for i in range(n_S)]
        idx_M = [i for i in range(n_M)]
        idx_L = [i for i in range(n_L)]
        idx_A = [i for i in range(n_A)]

        if lds_idx is not None:
            assert all(idx >= 0 for idx in lds_idx), f"System loudspeaker indices in {self.room_name} go from 0 to {n_L}."
            assert max(lds_idx) <= n_L, f"Only {n_L} system loudspeakers are available in {self.room_name}, but {max(lds_idx)+1}-th loudspeaker was requested."
            assert len(lds_idx) == len(set(lds_idx)), f"Requested system loudspeakers are not unique."
            n_L = len(lds_idx)
            idx_L = lds_idx
        if mcs_idx is not None:
            assert all(idx >= 0 for idx in mcs_idx), f"System microphone indices in {self.room_name} go from 0 to {n_L-1}."
            assert max(mcs_idx) <= n_M, f"Only {n_M} system microphones are available in {self.room_name}, but index {max(mcs_idx)} microphone was requested."
            assert len(mcs_idx) == len(set(mcs_idx)), f"Requested system microphones are not unique."
            n_M = len(mcs_idx)
            idx_M = mcs_idx

        return n_S, n_M, n_L, n_A, idx_S, idx_M, idx_L, idx_A
    
    def __ems_rcs_positions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        stg_pos_all = self.low_level_info['StageAndAudience']['StageEmitters']['Position_m']
        mcs_pos_all = self.low_level_info['AudioSetup']['SystemReceivers']['Position_m']
        lds_pos_all = self.low_level_info['AudioSetup']['SystemEmitters']['Position_m']
        aud_pos_all = self.low_level_info['StageAndAudience']['AudienceReceivers']['MonochannelPosition_m']

        stg_pos = []
        mcs_pos = []
        lds_pos = []
        aud_pos = []

        for s in self.idx_S:
            stg_pos.append(stg_pos_all[s])
        for m in self.idx_M:
            mcs_pos.append(mcs_pos_all[m])
        for l in self.idx_L:
            lds_pos.append(lds_pos_all[l])
        for a in self.idx_A:
            aud_pos.append(aud_pos_all[a])

        return stg_pos, mcs_pos, lds_pos, aud_pos

    def __load_rirs(self, ds_dir: str) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter, int]:
        r"""
        Loads all the room impulse responses from the dataset and returns them in processing modules.

            **Args**:
                - ds_dir (str): Path to the dataset.
            
            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage sources and audience positions.
                - dsp.Filter: Room impulse responses bewteen stage sources and system microphones.
                - dsp.Filter: Room impulse responses bewteen system loudspeakers and audience positions.
                - dsp.Filter: Room impulse responses bewteen system loudspeakers and system microphones.
                - int: Length of the room impulse responses in samples.
        """

        rir_info = self.low_level_info['RoomImpulseResponses']

        rir_fs = rir_info['SampleRate_Hz']
        rir_length = rir_info['LengthInSamples']

        if rir_fs != self.fs:
            rir_length = int(self.fs * rir_length/rir_fs)

        ds_dir = ds_dir.rstrip('/')
        path_root = f"{ds_dir}/data/{self.high_level_info['RoomDirectory']}/{rir_info['Directory']}"

        # Load RIRs
        path = f"{path_root}/{rir_info['StageEmitters-AudienceReceivers']['Directory']}"
        stg_to_aud = self.__load_rir_matrix(path=f"{path}", emitter_idx=self.idx_S, receiver_idx=self.idx_A, fs=rir_fs, n_samples=rir_length)
        path = f"{path_root}/{rir_info['StageEmitters-SystemReceivers']['Directory']}"
        stg_to_sys = self.__load_rir_matrix(path=f"{path}", emitter_idx=self.idx_S, receiver_idx=self.idx_M, fs=rir_fs, n_samples=rir_length)
        path = f"{path_root}/{rir_info['SystemEmitters-AudienceReceivers']['Directory']}"
        sys_to_aud = self.__load_rir_matrix(path=f"{path}", emitter_idx=self.idx_L, receiver_idx=self.idx_A, fs=rir_fs, n_samples=rir_length)
        path = f"{path_root}/{rir_info['SystemEmitters-SystemReceivers']['Directory']}"
        sys_to_sys = self.__load_rir_matrix(path=f"{path}", emitter_idx=self.idx_L, receiver_idx=self.idx_M, fs=rir_fs, n_samples=rir_length)

        # Energy normalization
        stg_to_aud_norm, stg_to_sys_norm, sys_to_aud_norm, sys_to_sys_norm = self.__normalize_rirs(
            stg_to_aud=stg_to_aud,
            stg_to_sys=stg_to_sys,
            sys_to_aud=sys_to_aud,
            sys_to_sys=sys_to_sys
        )

        # Create processing modules
        h_SA, h_SM, h_LA, h_LM = self.create_modules(
            rirs_SA=stg_to_aud_norm,
            rirs_SM=stg_to_sys_norm,
            rirs_LA=sys_to_aud_norm,
            rirs_LM=sys_to_sys_norm,
            rir_length=rir_length
        )

        return h_SA, h_SM, h_LA, h_LM, rir_length
    
    def __load_rir_matrix(self, path: str, emitter_idx: int, receiver_idx: int, fs: int, n_samples: int) -> torch.Tensor:
        r"""
        Loads the room impulse responses from the dataset and returns them in a matrix.

            **Args**:
                - path (str): Path to the room impulse responses in the dataset.
                - emitter_idx (list[int]): Indices of the emitters.
                - receiver_idx (list[int]): Indices of the receivers.
                - fs (int): Sample rate [Hz].
                - n_samples (int): Length of the room impulse responses in samples.

            **Returns**:
                - torch.Tensor: Room-impulse-response matrix as a torch tensor [n_samples, n_receivers, n_emitters].
        """
        n_emitters = len(emitter_idx)
        n_receivers = len(receiver_idx)

        matrix = torch.zeros(n_samples, n_receivers, n_emitters)
        for i,r in enumerate(receiver_idx):
            for j,e in enumerate(emitter_idx):
                w = torchaudio.load(f"{path}/E{e+1:03d}_R{r+1:03d}_M01.wav")[0]
                if self.fs != fs:
                    w = torchaudio.transforms.Resample(fs, self.fs)(w)
                matrix[:,i,j] = w.permute(1,0).squeeze()

        # Energy normalization
        # ec = energy_coupling(rir=matrix, fs=self.fs, decay_interval='T30')
        # norm_factor = torch.max(torch.tensor([self.n_L, self.n_M])) * torch.sqrt(torch.median(ec))
        # matrix = matrix/norm_factor

        return matrix

    def __normalize_rirs(self,
            stg_to_aud: torch.Tensor,
            stg_to_sys: torch.Tensor,
            sys_to_aud: torch.Tensor,
            sys_to_sys: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Normalizes the room impulse responses.
            **Args**:
                - stg_to_aud (torch.Tensor): Room impulse responses bewteen stage sources and audience positions.
                - stg_to_sys (torch.Tensor): Room impulse responses bewteen stage sources and system microphones.
                - sys_to_aud (torch.Tensor): Room impulse responses bewteen system loudspeakers and audience positions.
                - sys_to_sys (torch.Tensor): Room impulse responses bewteen system loudspeakers and system microphones.

            **Returns**:
                - torch.Tensor: Normalized room impulse responses bewteen stage sources and audience positions.
                - torch.Tensor: Normalized room impulse responses bewteen stage sources and system microphones.
                - torch.Tensor: Normalized room impulse responses bewteen system loudspeakers and audience positions.
                - torch.Tensor: Normalized room impulse responses bewteen system loudspeakers and system microphones.
        """
        # Energy couplings
        ec_sa = energy_coupling(rir=stg_to_aud, fs=self.fs, decay_interval='T30')
        ec_sm = energy_coupling(rir=stg_to_sys, fs=self.fs, decay_interval='T30')
        ec_la = energy_coupling(rir=sys_to_aud, fs=self.fs, decay_interval='T30')
        ec_lm = energy_coupling(rir=sys_to_sys, fs=self.fs, decay_interval='T30')

        # Normalization factors
        norm_stg = ( torch.sqrt(torch.median(ec_sa) * torch.median(ec_sm)) ) / ( torch.median(ec_sa) + torch.median(ec_sm) )
        norm_mcs = ( torch.sqrt(torch.median(ec_sm) * torch.median(ec_lm)) ) / ( torch.median(ec_sm) + torch.median(ec_lm) )
        norm_lds = ( torch.sqrt(torch.median(ec_la) * torch.median(ec_lm)) ) / ( torch.median(ec_la) + torch.median(ec_lm) )
        norm_aud = ( torch.sqrt(torch.median(ec_sa) * torch.median(ec_la)) ) / ( torch.median(ec_sa) + torch.median(ec_la) )

        # Normalization
        stg_to_aud = stg_to_aud / torch.sqrt(norm_stg * norm_aud)
        stg_to_sys = stg_to_sys / torch.sqrt(norm_stg * norm_mcs)
        sys_to_aud = sys_to_aud / torch.sqrt(norm_lds * norm_aud)
        sys_to_sys = sys_to_sys / torch.sqrt(norm_lds * norm_mcs)

        return stg_to_aud, stg_to_sys, sys_to_aud, sys_to_sys
