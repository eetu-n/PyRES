import torch 
from PyRES.plots import plot_irs_compare, plot_spectrograms_compare
from PyRES.rds import RDS

#torch.manual_seed(141122)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.set_default_device(device)
    #print(torch.get_default_device())

    fs = 48000

    rds = RDS(
        fs = fs,
        FIR_order = 2**16,
        expansion = 2**10,
        epochs = 10,
        room_name = "MarsioExperimentalStudio3MicSetup2",
        #lds_idx = list(range(0,11,2)),
        lds_idx = [0, 5],
        mcs_idx = [2, 3],
        device = torch.get_default_device()
    )

    print("Training started...")
    rds.train()
    print("Training ended.")
    # ------------------------- Plots -------------------------
    sys_nat,_,sys_full_opt = rds.res.system_simulation()

    #rds.res.phroom = alt_room

    #_,_,alt_sys = rds.res.system_simulation()

    #saved_file = rds.res.save_state_to("results")
    
    # ------------------------- Plots -------------------------
    plot_irs_compare(sys_nat, sys_full_opt, fs)
    #plot_irs_compare(sys_full_opt, alt_sys, fs)
    #plot_spectrograms_compare(sys_nat, sys_full_opt, samplerate, nfft = 2**9)
