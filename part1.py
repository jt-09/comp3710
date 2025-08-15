import torch
import numpy as np
import matplotlib.pyplot as plt

# show pytorch version
print("PyTorch Version:", torch.__version__)

# device config stuff (CUDA for gpu, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create coord grid
# np.mgrid creates two 2D arrays:
#   X contains the x-coordinate for each point
#   Y contains the y-coordinate for each point
# coords range from -4.0 to 4.0 with a step of 0.01
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into pytorch sensors
# in pyTorch:
#   - A tensor is like a NumPy array, but it can live on the GPU
#   - Tensors can be part of a computational graph (used in deep learning)
x = torch.Tensor(X).to(device) # move to device (GPU or CPU)
y = torch.Tensor(Y).to(device)

# Compute Gaussian: e^(-(x^2 + y^2) / 2)
# - Produces a smooth "hill" shape centered at (0,0)
z = torch.exp(-(x**2 + y**2) / 2.0)

# Plot the result
plt.imshow(z.cpu().numpy()) # Move to CPU & convert to NumPy before plotting
plt.tight_layout()
plt.show()



# numpy + matplotlib version
# Compute Gaussian: e^(-(x^2 + y^2) / 2)
Z = np.exp(-(X**2 + Y**2) / 2.0)

# Plot the Gaussian
plt.imshow(Z, extent=(-4, 4, -4, 4))
plt.title("2D Gaussian (NumPy)")
plt.show()






