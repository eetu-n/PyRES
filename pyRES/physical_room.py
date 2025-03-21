# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
from collections import OrderedDict
import json
# Torch
import torch
import torchaudio
# Flamo
from flamo import dsp


# ==================================================================
# ========================= ABSTRACT CLASS =========================

class PhRoom(object):
    r"""
    Physical room abstraction class.
    """
    def __init__(
            self,
            fs: int,
            nfft: int,
            alias_decay_db: float
        ) -> None:
        r"""
        Initializes the PhRoom object.
        """
        object.__init__(self)

        self.fs = fs
        self.nfft = nfft
        self.alias_decay_db = alias_decay_db

        self.n_S = None
        self.n_L = None
        self.n_M = None
        self.n_A = None

        self.rirs = torch.Tensor()
        self.rir_length = None
    
    def get_ems_rcs_number(self) -> OrderedDict:
        r"""
        Returns the number of emitters and receivers.

            **Returns**:
                OrderedDict: Number of sources and receivers.
        """
        return self.n_S, self.n_M, self.n_L, self.n_A
    
    def get_stg_to_aud(self) -> torch.Tensor:
        r"""
        Returns the sources to audience RIRs

            **Returns**:
                torch.Tensor: Sources to audience RIRs. shape = (samples, n_A, n_S).
        """
        return self.rirs['stg_to_aud']

    def get_stg_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the sources to microphones RIRs

            **Returns**:
                torch.Tensor: Sources to microphones RIRs. shape = (samples, n_M, n_S).
        """
        return self.rirs['stg_to_sys']
    
    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to audience RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to audience RIRs. shape = (samples n_A, n_L).
        """
        return self.rirs['sys_to_aud']

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to microphones RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to microphones RIRs. shape = (samples, n_M, n_L).
        """
        return self.rirs['sys_to_sys']
    
    def get_all_rirs(self) -> OrderedDict:
        r"""
        Returns a copy of the system RIRs.

            **Returns**:
                OrderedDict: The system room impulse responses.
        """
        RIRs = OrderedDict()
        RIRs.update({'H_SM': self.get_stg_to_mcs()})
        RIRs.update({'H_SA': self.get_stg_to_aud()})
        RIRs.update({'H_LM': self.get_lds_to_mcs()})
        RIRs.update({'H_LA': self.get_lds_to_aud()})
        return RIRs
    
    def create_modules(self) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter]:

        h_SM = dsp.Filter(
            size=(self.rir_length, self.n_M, self.n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_SM.assign_value(self.get_stg_to_mcs())

        h_SA = dsp.Filter(
            size=(self.rir_length, self.n_A, self.n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_SA.assign_value(self.get_stg_to_aud())

        h_LM = dsp.Filter(
            size=(self.rir_length, self.n_M, self.n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_LM.assign_value(self.get_lds_to_mcs())

        h_LA = dsp.Filter(
            size=(self.rir_length, self.n_A, self.n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_LA.assign_value(self.get_lds_to_aud())

        return h_SM, h_SA, h_LM, h_LA


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
            room_name: str
        ) -> None:
        r"""
        Initializes the PhRoom_dataset object.

            **Args**:
                - fs (int): Sample rate [Hz].
                - dataset_directory (str): Path to the dataset.
                - room_name (str): Name of the room.
        """
        super().__init__(
            self,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.high_level_info = self.__find_room_in_dataset(
            ds_dir=dataset_directory,
            room=room_name
        )

        self.low_level_info = self.__get_room_info(
            ds_dir=dataset_directory,
            room_dir=self.high_level_info['RoomDirectory']
        )

        self.n_S, self.n_M, self.n_L, self.n_A = self.__ems_rcs_number()
        self.rirs, self.rir_length = self.__load_rirs(
            ds_dir=dataset_directory
        )

        self.h_SM, self.h_SA, self.h_LM, self.h_LA = self.create_modules()

    def __find_room_in_dataset(self, ds_dir: str, room: str) -> dict:
        r"""
        Finds the room in the dataset.

            **Args**:
                - ds_dir (str): Path to the dataset.
                - room (str): Name of the room.
        """
        ds_dir = ds_dir.rstrip('/')
        with open(f"{ds_dir}/datasetInfo.json", 'r') as file:
            data = json.load(file)
        
        return data['Rooms'][room]
    
    def __get_room_info(self, ds_dir: str, room_dir: str) -> dict:
        r"""
        Gets the room information.
        """
        ds_dir = ds_dir.rstrip('/')
        with open(f"{ds_dir}/data/{room_dir}/roomInfo.json", 'r') as file:
            data = json.load(file)

        return data
    
    def __ems_rcs_number(self) -> tuple[int, int, int, int]:
        r"""
        Scans the room information for the number of emitters and receivers.
        """
        n_S = self.low_level_info['StageAndAudience']['StageEmitters']['Number']
        n_M = self.low_level_info['ActiveAcousticEnhancementSystem']['SystemReceivers']['Number']
        n_L = self.low_level_info['ActiveAcousticEnhancementSystem']['SystemEmitters']['Number']
        n_A = self.low_level_info['StageAndAudience']['AudienceReceivers']['MonochannelNumber']

        return n_S, n_M, n_L, n_A

    def __load_rirs(self, ds_dir: str) -> tuple[OrderedDict[str, torch.Tensor], int]:
        r"""
        Loads the room impulse responses.

            **Args**:
                - ds_dir (str): Path to the dataset.
            
            **Returns**:
                tuple[OrderedDict[str, torch.Tensor], int]: Room impulse responses and their length.
        """

        rirs_info = self.low_level_info['Rirs']

        rirs_fs = rirs_info['SampleRate_Hz']
        rirs_length = rirs_info['LengthInSamples']

        if rirs_fs != self.fs:
            rirs_length = int(self.fs * rirs_length/rirs_fs)

        ds_dir = ds_dir.rstrip('/')
        path_root = f"{ds_dir}/data/{self.high_level_info['RoomDirectory']}/{rirs_info['Directory']}"

        path = f"{path_root}/{rirs_info['StageAudienceRirs']['Directory']}"
        stg_to_aud = self.__load_rir_matrix(path=f"{path}", n_sources=self.n_S, n_receivers=self.n_A, fs=rirs_fs, n_samples=rirs_length)
        path = f"{path_root}/{rirs_info['StageSystemRirs']['Directory']}"
        stg_to_sys = self.__load_rir_matrix(path=f"{path}", n_sources=self.n_S, n_receivers=self.n_M, fs=rirs_fs, n_samples=rirs_length)
        path = f"{path_root}/{rirs_info['SystemAudienceRirs']['Directory']}"
        sys_to_aud = self.__load_rir_matrix(path=f"{path}", n_sources=self.n_L, n_receivers=self.n_A, fs=rirs_fs, n_samples=rirs_length)
        path = f"{path_root}/{rirs_info['SystemSystemRirs']['Directory']}"
        sys_to_sys = self.__load_rir_matrix(path=f"{path}", n_sources=self.n_L, n_receivers=self.n_M, fs=rirs_fs, n_samples=rirs_length)

        rirs = OrderedDict([
            ('stg_to_aud', stg_to_aud),
            ('stg_to_sys', stg_to_sys),
            ('sys_to_aud', sys_to_aud),
            ('sys_to_sys', sys_to_sys)
        ])

        return rirs, rirs_length
    
    def __load_rir_matrix(self, path: str, n_sources: int, n_receivers: int, fs: int, n_samples: int) -> torch.Tensor:
        r"""
        Loads the room impulse responses.
        """
        matrix = torch.zeros(n_samples, n_receivers, n_sources)
        for i in range(n_receivers):
            for j in range(n_sources):
                w = torchaudio.load(f"{path}/E{j+1:03d}_R{i+1:03d}_M01.wav")[0]
                if self.fs != fs:
                    w = torchaudio.transforms.Resample(fs, self.fs)(w)
                matrix[:,i,j] = w.permute(1,0).squeeze()

        # TODO: apply here normalization

        return matrix
    

# ==================================================================
# =================== WHITE GAUSSIAN NOISE CLASS ===================

class PhRoom_rayleighDistributed(PhRoom):
    r"""
    Subclass of PhRoom that generates the room impulse responses of a shoebox room approximated to exponentially-decaying random sequences drawn by the Rayleigh distribution.
    """
    def __init__(
            self,
            room_size: tuple[float, float, float],
            room_RT: float,
            fs: int,
            nfft: int,
            alias_decay_db: float,
            n_S: int,
            n_L: int,
            n_A: int, 
            n_M: int
        ) -> None:
        r"""
        Initializes the PhRoom_wgn object.

            **Args**:
                - room_size (tuple[float, float, float]): Room size in meters.
                - n_S (int): Number of sources. Defaults to 1.
                - n_L (int): Number of loudspeakers. Defaults to 1.
                - n_M (int): Number of microphones. Defaults to 1.
                - n_A (int): Number of audience members. Defaults to 1.
                - fs (int): Sample rate [Hz].
        """
        assert n_S > 0, "Not enough stage sources."
        assert n_L > 0, "Not enough system loudspeakers."
        assert n_M > 0, "Not enough system microphones."
        assert n_A > 0, "Not enough audience receivers."

        super().__init__(
            self,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.RT = room_RT
        self.room_size = room_size
        
        self.n_S = n_S
        self.n_L = n_L
        self.n_M = n_M
        self.n_A = n_A

        self.rirs, self.rir_length = self.__generate_rirs()
        self.h_SM, self.h_SA, self.h_LM, self.h_LA = self.create_modules()

    def __generate_rirs(self) -> tuple[OrderedDict[str, torch.Tensor], int]:
        # TODO: generate exponentially-decaying white-Gaussian-noise sequences and zero pad them based on room size simulating transducers positioning
        # NOTE: Check torch.distributions.chi2.Chi2() and apply torch.sqrt() to obtain a Rayleigh distribution
        pass


class PhRoom_ISM(PhRoom):
    r"""
    Subclass of PhRoom that generates the room impulse responses of a shoebox room simulated through Image Source Method (ISM).
    Reference:
        pyroomacoustics:
        https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html
    """
    def __init__(
            self,
            room_size: tuple[float, float, float],
            room_RT: float,
            ISM_max_order: int,
            fs: int,
            nfft: int,
            alias_decay_db: float,
            n_S: int,
            n_L: int,
            n_A: int,
            n_M: int
        ) -> None:
        r"""
        Initializes the PhRoom_wgn object.

            **Args**:
                - room_size (tuple[float, float, float]): Room size in meters.
                - room_RT (float): Room reverberation time [s].
                - ISM_max_order (int): Maximum order of the Image Source Method.
                - n_S (int): Number of sources. Defaults to 1.
                - n_L (int): Number of loudspeakers. Defaults to 1.
                - n_M (int): Number of microphones. Defaults to 1.
                - n_A (int): Number of audience members. Defaults to 1.
                - fs (int): Sample rate [Hz].
        """
        assert n_S > 0, "Not enough stage sources."
        assert n_L > 0, "Not enough system loudspeakers."
        assert n_M > 0, "Not enough system microphones."
        assert n_A > 0, "Not enough audience receivers."

        super().__init__(
            self,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.RT = room_RT
        self.room_size = room_size
        self.max_order = ISM_max_order
        
        self.n_S = n_S
        self.n_L = n_L
        self.n_M = n_M
        self.n_A = n_A

        self.rirs, self.rir_length = self.__generate_rirs()
        self.h_SM, self.h_SA, self.h_LM, self.h_LA = self.create_modules()

    def __generate_rirs(self) -> tuple[OrderedDict[str, torch.Tensor], int]:

        # e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        # room = pra.ShoeBox(
        #     room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
        # )

        pass