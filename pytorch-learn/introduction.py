#%%
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# %%
# torch.tensor(data) creates a torch.Tensor object with the given data
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)

# %%
# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.tensor(M_data)
print(M)

# %%
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

# %%

# %%
# 3个4x5矩阵
x = torch.randn((3, 4, 5))
print(x)

# %%
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(x)
print(y)
print(z)

# %%
# one matrix 2x5
x1 = torch.randn(2, 5)
# one matrix 3x5
y1 = torch.randn(3, 5)
# one matrix 5x5 result by x1, y1
z1 = torch.cat([x1, y1], 0)
print(x1)
print(y1)
print(z1)

# %%
x2 = torch.randn(2, 3)
y2 = torch.randn(2, 5)
z2 = torch.cat([x2, y2], 1)
print(x2)
print(y2)
print(z2)

#%%
# reshaping tensors
x = torch.randn(2,3,5)
print(x)
print(x.view(2,15)) # reshape to 2 rows, 12 columns
# same as above. If one of the dimensions is -1, its size can be inferred
print(x.view(2,-1))

#%%
# tensor factory methods have a ''requires_grad'' flag
x = torch.tensor([1., 2., 3.], requires_grad = True)

# with requires_grand = True, you can still do all the operations you previously
# could
z = x + y
print(z)
# BUT z knows something extra
print(z.grad_fn)

#%%
# how dose that help up compute a gradient
# sum up all the entries in z
s = z.sum()
print(s)
print(s.grad_fn)
s.backward()
print(x.grad)
print(x)

#%%
# by default, user created tensors have ''requires_grad = False''
x = torch.randn(2,2)
y = torch.randn(2,2)
print(x.requires_grad, y.requires_grad)
z = x + y
# you can't backprop through z
print(z.grad_fn)

# ''.requires_grad_(...)'' change an existing tensor's ''requires_grad''
# flag in-place. the input flag default to ''True'' if not given
x = x.requires_grad_();
y = y.requires_grad_();
z = x + y
# z contines enough information to compute gradients, as we saw above
print(z.grad_fn)
print(z.requires_grad)

# now z has the computation history that relate itself to x and y
# can we just take its values, and detach it from its history
new_z = z.detach()
print(new_z.grad_fn)
print(z.grad_fn)
# ''z.detach()''  returns a tensor that shares the same storage as 'z'
# but with the computation history forgotten. 
# it dosen't know anything about how it was computed
# In essence, we have broken the tensor away from past history

#%%
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)