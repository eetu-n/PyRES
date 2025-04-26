# PyRES
### A Python library for Reverberation Enhancement System development and simulation.

---

PyRES is designed to interface with the related open-access dataset:

De Bortoli, G. M. (2025). DataRES: Dataset for research on Reverberation Enhancement Systems (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15165524

---

## Installation

Follow the instructions below to install **PyRES** and to set up a working environment.

1. Clone the repository
```shell
git clone https://github.com/GianMarcoDeBortoli/PyRES.git
cd pyres
```

2. Install Python 3.10:

Make sure you have Python installed on your system. The preferred Python version for PyRES is 3.10.

3. Set up the environment
- Automatic setup (it will use **Conda** if possible):
  - on **Windows**:
    ```shell
    bootstrap.bat
    ```
  - On **MacOS/Linux**:
    ```shell
    bash bootstrap.sh
    ```
- Manual Setup:
  - If you are using **Pip**:
    ```shell
    python3 -m venv pyres-env
    source pyres-env/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    ```
  - If you are using **Conda**:
    ```shell
    conda env create -f environment.yml
    ```
    or
    ```shell
    conda env create -f environment.yml --name venv-name
    ```
    to also choose the name of the environment.

### Note on libsndfile (System Library)

If you created the working environment using **Pip** only (i.e., you answered "n" to the bootstrap script prompt or you have done a manual setup with **Pip**), you must manually install the system library `libsndfile`:

- On macOS: Install it using [Homebrew](https://brew.sh/) with `brew install libsndfile`.
- On Linux: Install it via apt with `sudo apt install libsndfile1`.
- On Windows: you can download the libsndfile binaries from [libsndfile.org](http://www.mega-nerd.com/libsndfile/) or use a precompiled version like the one available from Gohlke’s website.

If you do not install `libsndfile`, you will encounter errors when using audio functionality.

---

## Tutorial

Please refer to the example files for a tutorial on how to use this library.
