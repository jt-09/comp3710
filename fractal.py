import torch
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_min, x_max = -2.0, 1.0
y_min, y_max = -2.0, 1.0


# image res (higher for more detail, but slower)
width = 1000
height = 1000

# higher iterations capture more detail, but take longer and use more memory
max_iter = 200

# calc  step size based on the coord range and res.
x_step = (x_max - x_min) / width
y_step = (y_max - y_min) / height



Y, X = np.mgrid[y_min:y_max:y_step, x_min:x_max:x_step]


c_real = torch.Tensor(X)
c_imag = torch.Tensor(Y)
cs = torch.complex(c_real, c_imag)

zs = torch.zeros_like(cs)

# The counter for the "escape time".
ns = torch.zeros_like(cs, dtype=torch.float32)

# Move the tensors to the specified device for parallel computation.
cs = cs.to(device)
zs = zs.to(device)
ns = ns.to(device)

print(f"Computing a {width}x{height} Burning Ship fractal with {max_iter} iterations...")



for i in range(max_iter):
    # The core Burning Ship iteration: z_n+1 = (|real(z_n)| + i|imag(z_n)|)^2 + c
    #  computed in parallel for every pixel at once
    # take the  absval of the real and imaginary parts of z_n
    zs_abs_real = torch.abs(zs.real)
    zs_abs_imag = torch.abs(zs.imag)

    # Reconstruct the complex number and apply the standard iteration formula.
    zs_ = torch.complex(zs_abs_real, zs_abs_imag) ** 2 + cs

    mag_sq = zs_.real * zs_.real + zs_.imag * zs_.imag
    not_diverged = mag_sq < 4.0

    # Increment the counter for points that have not yet diverged.
    ns += not_diverged.to(torch.float32)

    # Update the working variable for the next iteration.
    zs = zs_

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,10))
def processFractal(a):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()),  extent=[x_min, x_max, y_min, y_max])
plt.tight_layout(pad=0)
plt.show()

