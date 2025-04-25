# PyRES
### Python library for Reverberation Enhancement Systems

---

PyRES is designed to interface with the related open-access dataset:

De Bortoli, G. M. (2025). DataRES: Dataset for research on Reverberation Enhancement Systems (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15165524

---

# Installation

PyRES relies on [flamo](https://github.com/gdalsanto/flamo) as the back end.

Further dependencies are:
- pyfar
- pyrato
- seaborn

To install via conda:
```shell
conda create -n pyres-env python=3.10
conda activate pyres-env
pip install flamo
conda install -c conda-forge libsndfile
pip install pyfar pyrato seaborn
```

---

# Tutorial

Please refer to the example files for a tutorial on how to use this library.
