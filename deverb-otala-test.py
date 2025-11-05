import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import torch.nn as nn
#import loss

from loss import ESRLoss

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo_trainer import Trainer

import flamo.optimize.loss as loss
from flamo_loss import edc_loss

from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.plots import plot_irs_compare, plot_spectrograms_compare
from PyRES.loss_functions import BruteForceDirectPath, PunishHighValues

#torch.manual_seed(141122)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    samplerate = 48000
    nfft = samplerate
    alias_decay_db = 0.0
    FIR_order = 2**16
    lr = 0.1
    expansion = 2**12
    epochs = 30
    step_size = 200
    step_factor = 0.4

    # Physical room
    dataset_directory = './dataRES'
    room_name = 'Otala'
    alt_room_name = 'Otala_C1'
    #room_name = 'MarsioExperimentalStudio3MicSetup2'

    #train_dir = os.path.join('training_output', time.strftime("%Y%m%d-%H%M%S"))
    #os.makedirs(train_dir, exist_ok=True)

    # Loading Dataset 
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name,
        device = device
    )
    n_M = physical_room.transducer_number['mcs']  # Number of microphones
    n_L = physical_room.transducer_number['lds']  # Number of loudspeakers

    print(f"Number of Mics:", n_M)
    print(f"Number of Loudspeakers:", n_L)

    alt_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name = alt_room_name,
        device = device
    )

    # Using Random FIRs in the Virtual Room to train for deverb
    virtual_room = random_FIRs(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        FIR_order=FIR_order,
        requires_grad=True,
        device = device
    )

    # Init RES object 
    res = RES(physical_room=physical_room, virtual_room=virtual_room)

    #print("CUDA available:", torch.cuda.is_available())
    #print("Device count:", torch.cuda.device_count())
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

    model.to(device)

    sys_nat,_,_ = res.system_simulation()

    #print(sys_nat.shape)
    
    dataset_target = torch.zeros(1, samplerate, 1, device=device)
    #dataset_target[:,1990,:] = 1 # Delayed to the prop delay from the loudspeaker. 
    #dataset_target[0,0:2050,0] = sys_nat[0:2050,0]
    dataset_target[0,0:290,0] = sys_nat[0:290,0]
    dataset_target.to(device)

    dataset_input = torch.zeros(1, samplerate, 1, device=device)
    dataset_input[:,0,:] = 1
    dataset_input.to(device)
    # dataset_input = sys_nat.permute(0, 1).unsqueeze(0)
    #print(f"Input Dataset Shape:", dataset_input.shape)
    #print(f"Target Dataset Shape:", dataset_target.shape)
    
    dataset = Dataset(
        input = dataset_input,
        target = dataset_target,
        expand = expansion,
        device = device
    )
    
    train_loader, valid_loader = load_dataset(dataset, batch_size=1, split=0.8, shuffle=False, device=device)

    trainer = Trainer(
        net=model,
        max_epochs = epochs,
        lr = lr,
        patience_delta = 0.01,
        patience = 5,
        step_size = step_size,
        step_factor = step_factor,
        #train_dir = train_dir,
        device = device
    )

    mse = loss.mse_loss(nfft=nfft, device=device)
    esr = ESRLoss()

    edc = edc_loss(
        sample_rate = samplerate,
        is_broadband = False,
        n_fractions = 1,
        energy_norm = False,
        convergence = True,
        clip = False,
        device = device
    )

    apl = loss.AveragePower(device=device)

    bfd = BruteForceDirectPath()
    phv = PunishHighValues()

    #trainer.register_criterion(mse, 0.7 * 1000)
    trainer.register_criterion(esr, 2.0 * 1)
    #trainer.register_criterion(bfd, 10)
    trainer.register_criterion(phv, 10)

    print("Training started...")
    trainer.train(train_loader, valid_loader)
    print("Training ended.")

    _,_,sys_full_opt = res.system_simulation()

    res.phroom = alt_room

    _,_,alt_sys = res.system_simulation()

    saved_file = res.save_state_to("results")
    
    # ------------------------- Plots -------------------------
    plot_irs_compare(sys_nat, sys_full_opt, samplerate)
    plot_irs_compare(sys_full_opt, alt_sys, samplerate)
    #plot_spectrograms_compare(sys_nat, sys_full_opt, samplerate, nfft = 2**9)
