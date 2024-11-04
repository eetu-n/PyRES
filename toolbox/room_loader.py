from collections import OrderedDict
import json

import torch
import torchaudio

class AA_RIRs(object):
    def __init__(self, room_name: str, n_S: int, n_L: int, n_M: int, n_A: int, fs: int) -> None:
        r"""
        Room impulse response wrapper class.
        These room impulse responses were measured in the listening room called Otala inside
        the Aalto Acoustics Lab in the Aalto University's Otaniemi campus, Espoo, Finland.

            **Args**:
                - dir (str): Path to the room impulse responses.
                - n_S (int): Number of sources. Defaults to 1.
                - n_L (int): Number of loudspeakers. Defaults to 1.
                - n_M (int): Number of microphones. Defaults to 1.
                - n_A (int): Number of audience members. Defaults to 1.
                - fs (int): Sample rate [Hz].
        """
        object.__init__(self)
        assert n_S == 1, "Only one source is supported."
        assert n_L <= 13, "Only up to 13 loudspeakers are supported."
        assert n_M <= 4, "Only up to 4 microphones are supported."
        assert n_A == 1, "Only one audience member is supported."
        
        self.n_S = n_S
        self.n_L = n_L
        self.n_M = n_M
        self.n_A = n_A
        self.fs = fs
        self.room_data = self.__find_room_in_dataset(room_name)
        self.__RIRs, self.RIR_length = self.__load_rirs()

    def __find_room_in_dataset(self, room_name) -> str:
        r"""
        Finds the room in the dataset.
        """
        with open(f"./AA_Dataset/datasetInfo.json", 'r') as file:
            data = json.load(file)

        
        return data['Rooms'][room_name]

    def __load_rirs(self) -> tuple[OrderedDict[str, torch.Tensor], int]:
        r"""
        Loads the room impulse responses.

            **Returns**:
                tuple[OrderedDict[str, torch.Tensor], int]: Room impulse responses and their length.
        """

        # Replace 'your_file.json' with the path to your JSON file
        room_dir_inside_dataset = self.room_data['RoomDirectory']

        with open(f"AA_Dataset/data/{room_dir_inside_dataset}/roomInfo.json", 'r') as file:
            data = json.load(file)

        rirs_fs = data['Rirs']['SampleRate_Hz']
        rirs_length = data['Rirs']['RirsLengthInSamples']

        new_rirs_length = int(self.fs * rirs_length/rirs_fs) # I should infer this from the resample

        src_to_aud = torch.zeros(new_rirs_length, self.n_A, self.n_S)
        for i in range(self.n_A):
            for j in range(self.n_S):
                w = torchaudio.load(f"{self.dir}/StageAudience/R{i+1:03d}_E{j+1:03d}.wav")[0]
                if self.fs != rirs_fs:
                    w = torchaudio.transforms.Resample(rirs_fs, self.fs)(w)
                src_to_aud[:,i,j] = w.permute(1,0).squeeze()

        src_to_sys = torch.zeros(new_rirs_length, self.n_M, self.n_S)
        for i in range(self.n_M):
            for j in range(self.n_S):
                w = torchaudio.load(f"{self.dir}/StageSystem/R{i+1:03d}_E{j+1:03d}.wav")[0]
                if self.fs != rirs_fs:
                    w = torchaudio.transforms.Resample(rirs_fs, self.fs)(w)
                src_to_sys[:,i,j] = w.permute(1,0).squeeze()

        sys_to_aud = torch.zeros(new_rirs_length, self.n_A, self.n_L)
        for i in range(self.n_A):
            for j in range(self.n_L):
                w = torchaudio.load(f"{self.dir}/SystemAudience/R{i+1:03d}_E{j+1:03d}.wav")[0]
                if self.fs != rirs_fs:
                    w = torchaudio.transforms.Resample(rirs_fs, self.fs)(w)
                sys_to_aud[:,i,j] = w.permute(1,0).squeeze()

        sys_to_sys = torch.zeros(new_rirs_length, self.n_M, self.n_L)
        for i in range(self.n_M):
            for j in range(self.n_L):
                w = torchaudio.load(f"{self.dir}/SystemSystem/R{i+1:03d}_E{j+1:03d}.wav")[0]
                if self.fs != rirs_fs:
                    w = torchaudio.transforms.Resample(rirs_fs, self.fs)(w)
                sys_to_sys[:,i,j] = w.permute(1,0).squeeze()

        rirs = OrderedDict([
            ('src_to_aud', src_to_aud),
            ('src_to_sys', src_to_sys),
            ('sys_to_aud', sys_to_aud),
            ('sys_to_sys', sys_to_sys)
        ])

        return rirs, new_rirs_length
    
    def get_scs_to_aud(self) -> torch.Tensor:
        r"""
        Returns the sources to audience RIRs

            **Returns**:
                torch.Tensor: Sources to audience RIRs. shape = (15000, n_A, n_S).
        """
        return self.__RIRs['src_to_aud']

    def get_scs_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the sources to microphones RIRs

            **Returns**:
                torch.Tensor: Sources to microphones RIRs. shape = (15000, n_M, n_S).
        """
        return self.__RIRs['src_to_sys']
    
    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to audience RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to audience RIRs. shape = (15000, n_A, n_L).
        """
        return self.__RIRs['sys_to_aud']

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to microphones RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to microphones RIRs. shape = (15000, n_M, n_L).
        """
        return self.__RIRs['sys_to_sys']