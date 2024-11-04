def AA_dafx24_test(args):

    # Model
    FIR_order = 100
    model = AA_dafx24(
        n_S = 1,
        n_M = args.input_channels,
        n_L = args.output_channels,
        n_A = 1,
        fs = args.samplerate,
        nfft = args.nfft,
        FIR_order = FIR_order
        )
    reverb_RT = 0.5
    model.add_WGN(reverb_RT)
    
    # save initialization response
    with torch.no_grad():
        evs_init = get_magnitude(model.get_F_MM_eigenvalues())
        y_init = model.system_simulation()
    # save_audio(os.path.join(args.train_dir, 'y_init.wav'), y_init[0, :], args.samplerate)

    # Initialize dataset
    dataset = Dataset_Colorless(
        in_shape = (args.nfft//2+1, model.n_M),
        target_shape = (args.nfft//2+1, model.n_M),
        ds_len = args.num,
        device = args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=args.batch_size, split=args.split, shuffle=args.shuffle)

    # Initialize training process
    criterion = MSE_evs(args.num, args.nfft//2+1)
    trainer = Trainer(model, args.max_epochs, args.lr, args.device)
    trainer.register_criterion(criterion, 1)
    # Train the model
    trainer.train(train_loader, valid_loader)

    # save optimized response
    with torch.no_grad():
        evs_optim = get_magnitude(model.get_F_MM_eigenvalues())
        y_optim = model.system_simulation()
    save_audio(os.path.join(args.train_dir, 'y_out.wav'), y_optim[0, :], args.samplerate)

    plt.figure()
    plt.plot(mag2db(evs_init[0, :]).detach().numpy(), label='Initial')
    plt.plot(mag2db(evs_optim[0, :]).detach().numpy(), label='Optimized')
    plt.legend()

    plt.figure()
    plt.plot(y_init[0, :].detach().numpy(), label='Initial')
    plt.plot(y_optim[0, :].detach().numpy(), label='Optimized')
    plt.legend()

    plt.figure()
    plt.subplot(2,1,1)
    plt.specgram(y_init[0,:].detach().squeeze().numpy())
    plt.subplot(2,1,2)
    plt.specgram(y_optim[0,:].detach().squeeze().numpy())

    evs = torch.cat((evs_init.unsqueeze(3), evs_optim.unsqueeze(3)), (-1))

    model.plot_evs(mag2db(evs[:,20:20000,:,:]))
    return None