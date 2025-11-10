import h5py, numpy as np
print("h5py=", h5py.__version__)
print("hdf5_lib_version=", h5py.version.hdf5_version)
import numpy
print("numpy=", numpy.__version__)
try:
    import torch
    print("torch=", torch.__version__)
except Exception as e:
    print("torch import fail:", e)
