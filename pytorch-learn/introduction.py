import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# torch.tensor(data) creates a torch.Tensor object with the given data
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)

# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.tensor(M_data)
print(M)

# create a 3D tensor of size 2x2x2
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]
          ]

T = torch.tensor(T_data)
print(T)

print(V[0])
print(V[0].item())
print(M[0])
print(T[0])

x = torch.randn((3,4,5))
print(x)

x = torch.tensor([1. ,2. ,3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(x)
print(y)
print(z)
