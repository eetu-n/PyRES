import torch

from PyRES.loss_functions import ESRLoss
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.res import RES
from PyRES.utils import find_direct_path

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo_trainer import Trainer #TODO: Patch flamo


#TODO: Documentation etc.
# The philosophy here is that this should be a class someone can just
# instantiate with a room dataset and then train without having to know the
# internals of RES or whatever. The arguments are things you might want to
# change per room etc. Still needs work but it's easier to work with for now?
# I THINK
class RDS(object):
    def __init__(
            self,

            fs: int = 48000,
            nfft: int = 48000,
            alias_decay_db: float = 0.0,
            FIR_order: int = 2**16,
            lr: float = 0.1,
            expansion: int = 2**12,
            epochs: int = 30,
            step_size: int = 200,
            step_factor: float = 0.4,

            train_dir: str = None,

            dataset_directory: str = './dataRES',
            room_name: str = 'Otala',

            stg_idx: list[int] = None,
            mcs_idx: list[int] = None,
            lds_idx: list[int] = None,
            aud_idx: list[int] = None,

            device: torch.device = torch.get_default_device()

        ) -> None:
        # TODO: Split this up into functions that make sense
        object.__init__(self)

        self.device = device

        self.fs = fs
        self.nfft = nfft
        
        self.physical_room = PhRoom_dataset(
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db,
            dataset_directory=dataset_directory,
            room_name=room_name,
            stg_idx = stg_idx,
            mcs_idx = mcs_idx,
            lds_idx = lds_idx,
            aud_idx = aud_idx,
            device = device
        )

        self.num_mics = self.physical_room.transducer_number['mcs']
        self.num_speakers = self.physical_room.transducer_number['lds']

        self.virtual_room = random_FIRs(
            n_M = self.num_mics,
            n_L = self.num_speakers,
            fs = fs,
            nfft = nfft,
            alias_decay_db = alias_decay_db,
            FIR_order = FIR_order,
            requires_grad = True,
            device = self.device
        )

        self.res = RES(self.physical_room, self.virtual_room)

        self.model = system.Shell(
            core = self.res.full_system_(),
            input_layer = dsp.FFT(nfft=nfft),
            output_layer = dsp.iFFT(nfft=nfft)
        )

        natural_system_response, _, _ = self.res.system_simulation()

        self.dirct_path_delay = find_direct_path(natural_system_response[:,0]) + int(fs * 0.001) #TODO: Make this not dumb

        target = torch.zeros(1, fs, 1) #TODO: Always one second?
        target[0,0:self.dirct_path_delay,0] = natural_system_response[0:self.dirct_path_delay,0]

        input = torch.zeros(1, fs, 1) #TODO: Always this shape?
        input[:,0,:] = 1

        dataset = Dataset(
            input = input,
            target = target,
            expand = expansion,
            device = device
        )

        self.train_loader, self.valid_loader = load_dataset(dataset, batch_size=1, split=0.9, shuffle=False)

        self.trainer = Trainer(
            net=self.model,
            max_epochs = epochs,
            lr = lr,
            patience_delta = 0.005,
            patience = 3,
            step_size = step_size,
            step_factor = step_factor,
            train_dir = train_dir,
            device = device
        )

        self.trainer.register_criterion(ESRLoss(), 2)

        pass
    
    def train(self):
        self.trainer.train(self.train_loader, self.valid_loader)