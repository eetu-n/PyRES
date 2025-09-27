
if __name__ == '__main__':

    # Time-frequency
    samplerate = 48000              # Sampling frequency
    nfft = samplerate*3             # FFT size
    alias_decay_db = 0              # Anti-time-aliasing decay in dB

    # Physical room
    print(f"\nPhRoom_dataset interfaces with DataRES loading number and positions of all transducers and all RIRs between them.")
    print(f"PhRoom_dataset need to be gi