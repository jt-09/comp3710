############################################################################
# BURNING SHIP #############################################################
############################################################################

import torch
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_min, x_max = -2.0, 1.0
y_min, y_max = -2.0, 1.0

# higher iterations capture more detail, but take longer and use more memory
max_iter = 200


Y, X = np.mgrid[y_min:y_max:0.005, x_min:x_max:0.005]


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
fig = plt.figure(figsize=(10,6))
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

############################################################################
# NEWTONS FRATCAL ##########################################################
############################################################################

# # import torch
# import numpy as np
# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# x_min, x_max = -1.5, 1.5
# y_min, y_max = -1.5, 1.5

# # higher iterations capture more detail, but take longer and use more memory
# max_iter = 200

# Y, X = np.mgrid[y_min:y_max:0.005, x_min:x_max:0.005]

# # load into PyTorch tensors
# x = torch.Tensor(X)
# y = torch.Tensor(Y)
# z = torch.complex(x, y)          # initial z for newtons fractal
# zs = z.clone()                   # working variable that will hold z_n
# ns = torch.zeros_like(x)         # counter/"escape time" proxy

# # transfer to the GPU device
# z = z.to(device)
# zs = zs.to(device)
# ns = ns.to(device)



# # --- NEWTON FRACTAL: (f(z)=z^3 - 1) ---
# tol = 1e-6                       # convergence tolerance
# eps = torch.tensor(1e-12, device=device)  # derivative safety

# for i in range(max_iter):
#     # f = zs*zs*zs - 2*zs + 2
#     # f_dash = 3*zs*zs - 2
#     f = zs*zs*zs - 1
#     f_dash = 3*zs*zs 

#     # tiny bias in the denominator to avoid division by 0
#     zs = zs - f / (f_dash + eps)

#     # Count iterations until converged
#     not_converged = torch.abs(f) > tol
#     #Update variables to compute
#     ns += not_converged.to(ns.dtype)

# #plot

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(16,10))
# def processFractal(a):
#     """Display an array of iteration counts as a
#     colorful picture of a fractal."""
#     a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
#     img = np.concatenate([10+20*np.cos(a_cyclic),
#     30+50*np.sin(a_cyclic),
#     155-80*np.cos(a_cyclic)], 2)
#     img[a==a.max()] = 0
#     a = img
#     a = np.uint8(np.clip(a, 0, 255))
#     return a

# # plt.imshow(processFractal(ns.cpu().numpy()),  extent=[x_min, x_max, y_min, y_max])
# plt.imshow(ns.cpu().numpy(),  extent=[x_min, x_max, y_min, y_max],  cmap='inferno')

# plt.tight_layout(pad=0)
# plt.show()

