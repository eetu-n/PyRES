import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import torch.nn as nn
import loss

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.optimize.loss import mss_loss

from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.plots import plot_irs_compare

torch.manual_seed(141122)

class ThresholdedEDCLoss(nn.Module):
    def __init__(self, threshold_db):
        super().__init__()
        self.threshold_linear = 10**(threshold_db / 20)
    
    def forward(self, output, target):
        output_thresh = torch.where(torch.abs(output) < self.threshold_linear, 
                                torch.zeros_like(output), output)
        target_thresh = torch.where(torch.abs(target) < self.threshold_linear, 
                                torch.zeros_like(target), target)
        
        return loss.EDCLoss()(output_thresh, target_thresh)

if __name__ == '__main__':
    samplerate = 48000
    nfft = samplerate
    alias_decay_db = 0.0
    FIR_order = 2**18
    lr = 1e-3 
    epochs = 10
    threshold_db = -40 # Adjust according to noise floor

    # Physical room
    dataset_directory = './dataRES'
    room_name = 'MarsioExperimentalStudio3_P1'

    train_dir = os.path.join('training_output', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(train_dir, exist_ok=True)

    # Loading Dataset 
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )
    n_M = physical_room.transducer_number['mcs']  # Number of microphones
    n_L = physical_room.transducer_number['lds']  # Number of loudspeakers

    print(f"Number of Mics:", n_M)
    print(f"Number of Loudspeakers:", n_L)

    # Using Random FIRs in the Virtual Room to train for deverb
    virtual_room = random_FIRs(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        FIR_order=FIR_order,
        requires_grad=True
    )

    # Init RES object 
    res = RES(physical_room=physical_room, virtual_room=virtual_room)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU.")

    # Init Model
    # Full system loop, which is closed loop + possible transfer paths  
    model = system.Shell(
        core=res.full_system_(),
        input_layer=dsp.FFT(nfft=nfft),
        output_layer=dsp.iFFT(nfft=nfft)
    )

    sys_nat,_,_ = res.system_simulation()
    
    dataset_target = torch.zeros(1, samplerate, 1)
    dataset_target[:,1985,:] = 1 # Delayed to the prop delay from the loudspeaker. 

    dataset_input = torch.zeros(1, samplerate, 1)
    dataset_input[:,0,:] = 1
    # dataset_input = sys_nat.permute(0, 1).unsqueeze(0)
    print(f"Input Dataset Shape:", dataset_input.shape)
    print(f"Target Dataset Shape:", dataset_target.shape)
    
    dataset = Dataset(
        input = dataset_input,
        target = dataset_target,
        expand = 2**8,
        device = device
    )
    
    train_loader, valid_loader = load_dataset(dataset, batch_size=1, split=0.8, shuffle=False)

    trainer = Trainer(
        net=model,
        max_epochs = epochs,
        lr = lr,
        patience_delta = 10e-3,
        train_dir = train_dir,
        device = device
    )
    

    #criterion = loss.ScaledMSELoss()
    mss = mss_loss()
    esr = ThresholdedEDCLoss(threshold_db=threshold_db)

    trainer.register_criterion(esr, 1.0)

    print("Training started...")
    trainer.train(train_loader, valid_loader)
    print("Training ended.")

    _,_,sys_full_opt = res.system_simulation()
    
    # ------------------------- Plots -------------------------
    plot_irs_compare(sys_nat, sys_full_opt, samplerate)