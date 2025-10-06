import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer

from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.loss_functions import MSE_evs_mod
#from PyRES.functional import system_equalization_curve
from PyRES.plots import plot_evs_compare, plot_irs_compare

torch.manual_seed(141122)

if __name__ == '__main__':
    samplerate = 48000
    nfft = samplerate
    alias_decay_db = 0.0
    FIR_order = 2**8
    lr = 1e-3 
    epochs = 100

    # Physical room
    dataset_directory = './dataRES'
    room_name = 'Otala'

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
    n_L = physical_room.transducer_number['lds'] 

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

    # Init Model
    # needs full system loop, which is open loop + closed loop + possible transfer paths
    # 
    model = system.Shell(
        core=res.open_loop(),
        input_layer=system.Series(
            dsp.FFT(nfft=nfft),
            dsp.Transform(lambda x: x.diag_embed())
        )
    )

    evs_init = res.open_loop_eigenvalues()
    sys_nat,_,_ = res.system_simulation()

    dataset_target = torch.zeros(1, samplerate, n_M)
    dataset_target[:,0,:] = 1

    dataset_input = torch.zeros(1, samplerate, n_M)
    dataset_input[:,0,:] = 1
    
    print(f"Input Dataset Shape:", dataset_input.shape)
    print(f"Target Dataset Shape:", dataset_target.shape)
    
    dataset = Dataset(
        input = dataset_input,
        target = dataset_target,
        expand = samplerate,
        device = device
        )
    
    train_loader, valid_loader = load_dataset(dataset, batch_size=1, split=0.8, shuffle=False)

    trainer = Trainer(
        net=model,
        max_epochs = epochs,
        lr = lr,
        patience = 20,
        patience_delta = 1e-5,
        step_size = 1000,
        train_dir = train_dir,
        device = device
    )
    # MSE with the system response and a unit impulse 
    criterion = MSE_evs_mod(
        iter_num = 2**5,
        freq_points = nfft//2+1,
        samplerate = samplerate,
        lowest_f = 20,
        highest_f = samplerate / 2
    )
    trainer.register_criterion(criterion, 1.0)

    trainer.train(train_loader, valid_loader)
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    evs_opt = res.open_loop_eigenvalues()
    sys_opt,_,sys_full_opt = res.system_simulation()
    
    # ------------------------- Plots -------------------------
    plot_evs_compare(evs_init, evs_opt, samplerate, nfft, 20, samplerate / 2)
    plot_irs_compare(sys_nat, sys_full_opt, samplerate)