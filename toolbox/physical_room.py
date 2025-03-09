from collections import OrderedDict
import json

import torch
import torchaudio

class PhRoom(object):
    def __init__(self, fs: int, room_name: str) -> None:
        r"""
        Room impulse response wrapper class.

            **Args**:
                - dir (str): Path to the room impulse responses.
                - n_S (int): Number of sources. Defaults to 1.
                - n_L (int): Number of loudspeakers. Defaults to 1.
                - n_M (int): Number of microphones. Defaults to 1.
                - n_A (int): Number of audience members. Defaults to 1.
                - fs (int): Sample rate [Hz].
        """
        object.__init__(self)

        self.fs = fs

        self.room_data = self.__find_room_in_dataset(room_name)

        self.n_S = self.room_data['n_S']
        self.n_L = self.room_data['n_L']
        self.n_M = self.room_data['n_M']
        self.n_A = self.room_data['n_A']

        self.__rirs, self.rir_length = self.__load_rirs()

    def __find_room_in_dataset(self, room_name) -> str:
        r"""
        Finds the room in the dataset.
        """
        # TODO: Also, the dataset is not necessarily next to the toolbox, where should it go? Do I just ask people to place here the path?
        with open(f"./AA_Dataset/datasetInfo.json", 'r') as file:
            data = json.load(file)
        
        return data['Rooms'][room_name]

    def __load_rirs(self) -> tuple[OrderedDict[str, torch.Tensor], int]:
        r"""
        Loads the room impulse responses.

            **Returns**:
                tuple[OrderedDict[str, torch.Tensor], int]: Room impulse responses and their length.
        """

        room_directory = self.room_data['RoomDirectory']

        with open(f"AA_Dataset/data/{room_directory}/roomInfo.json", 'r') as file:
            data = json.load(file)

        rirs_fs = data['Rirs']['SampleRate_Hz']
        rirs_length = data['Rirs']['RirsLengthInSamples']

        if rirs_fs != self.fs:
            rirs_length = int(self.fs * rirs_length/rirs_fs)

        stg_to_aud = self.__load_rir_matrix(path=f"{room_directory}/StageAudience", n_sources=self.n_S, n_receivers=self.n_A, fs=rirs_fs, n_samples=rirs_length)
        stg_to_sys = self.__load_rir_matrix(path=f"{room_directory}/StageSystem", n_sources=self.n_S, n_receivers=self.n_M, fs=rirs_fs, n_samples=rirs_length)
        sys_to_aud = self.__load_rir_matrix(path=f"{room_directory}/SystemAudience", n_sources=self.n_L, n_receivers=self.n_A, fs=rirs_fs, n_samples=rirs_length)
        sys_to_sys = self.__load_rir_matrix(path=f"{room_directory}/SystemSystem", n_sources=self.n_L, n_receivers=self.n_M, fs=rirs_fs, n_samples=rirs_length)

        rirs = OrderedDict([
            ('src_to_aud', stg_to_aud),
            ('src_to_sys', stg_to_sys),
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
                w = torchaudio.load(f"{path}/R{i+1:03d}_E{j+1:03d}.wav")[0]
                if self.fs != fs:
                    w = torchaudio.transforms.Resample(fs, self.fs)(w)
                matrix[:,i,j] = w.permute(1,0).squeeze()

        return matrix
    
    def get_number_source_receivers(self) -> OrderedDict:
        r"""
        Returns the number of sources and receivers.

            **Returns**:
                OrderedDict: Number of sources and receivers.
        """
        scs_rcs = OrderedDict()
        scs_rcs.update({'n_S': self.n_S})
        scs_rcs.update({'n_M': self.n_M})
        scs_rcs.update({'n_L': self.n_L})
        scs_rcs.update({'n_A': self.n_A})
        return scs_rcs
    
    def get_scs_to_aud(self) -> torch.Tensor:
        r"""
        Returns the sources to audience RIRs

            **Returns**:
                torch.Tensor: Sources to audience RIRs. shape = (samples, n_A, n_S).
        """
        return self.__rirs['src_to_aud'].clone().detach()

    def get_scs_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the sources to microphones RIRs

            **Returns**:
                torch.Tensor: Sources to microphones RIRs. shape = (samples, n_M, n_S).
        """
        return self.__rirs['src_to_sys'].clone().detach()
    
    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to audience RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to audience RIRs. shape = (samples n_A, n_L).
        """
        return self.__rirs['sys_to_aud'].clone().detach()

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to microphones RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to microphones RIRs. shape = (samples, n_M, n_L).
        """
        return self.__rirs['sys_to_sys'].clone().detach()
    
    def get_all_rirs(self) -> OrderedDict:
        r"""
        Returns a copy of the system RIRs.

            **Returns**:
                OrderedDict: The system room impulse responses.
        """
        RIRs = OrderedDict()
        RIRs.update({'H_SM': self.get_scs_to_mcs()})
        RIRs.update({'H_SA': self.get_scs_to_aud()})
        RIRs.update({'H_LM': self.get_lds_to_mcs()})
        RIRs.update({'H_LA': self.get_lds_to_aud()})
        return RIRs
    

class PhRoom_ideal(object, PhRoom):
    def __init__(self, fs: int, n_S: int, n_L: int, n_A: int, n_M: int) -> None:
        r"""
        Room impulse response wrapper class.

            **Args**:
                - dir (str): Path to the room impulse responses.
                - n_S (int): Number of sources. Defaults to 1.
                - n_L (int): Number of loudspeakers. Defaults to 1.
                - n_M (int): Number of microphones. Defaults to 1.
                - n_A (int): Number of audience members. Defaults to 1.
                - fs (int): Sample rate [Hz].
        """
        assert n_S >= 0, "Not enough stage sources."
        assert n_L >= 1, "Not enough system loudspeakers."
        assert n_M >= 1, "Not enough system microphones."
        assert n_A >= 0, "Not enough audience receivers."

        object.__init__(self)

        self.fs = fs
        
        self.n_S = n_S
        self.n_L = n_L
        self.n_M = n_M
        self.n_A = n_A

        self.__rirs, self.rir_length = self.__generate_rirs()

    def __generate_rirs(self):
        pass
