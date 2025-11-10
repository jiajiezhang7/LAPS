import numpy as np, torch
x = np.zeros((2,3), dtype=np.uint8)
y = torch.from_numpy(x)
print('OK', y.shape, y.dtype)
