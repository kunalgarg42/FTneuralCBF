import torch.distributions as td
import torch
import matplotlib.pyplot as plt

n_dims = 2
batch = 1000

# 1: Choose which normal vector determines the plane.
normal_idx = torch.randint(0, n_dims, size=(batch, ))
assert normal_idx.shape == (batch,)

# 2: Choose whether it takes the value of hi or lo.
direction = torch.randint(2, size=(batch,), dtype=torch.bool)
assert direction.shape == (batch,)

lo = torch.Tensor([-1, -2])
hi = torch.Tensor([4, 5])
assert lo.shape == hi.shape == (n_dims, )
dist = td.Uniform(lo, hi)

samples = dist.sample((batch, ))
assert samples.shape == (batch, n_dims)


tmp = torch.where(direction, hi[normal_idx], lo[normal_idx])
print(tmp.shape)
assert tmp.shape == (batch,)

# print(tmp.shape)
# tmp = 13 * torch.ones(batch)
tmp = tmp[:, None].repeat(1, n_dims)

# print("samples")
# print(samples)
# print("samples2")
# print(samples[:, normal_idx])
# print(samples[:, normal_idx].shape)

# tmp2 = torch.arange(batch * n_dims).reshape((batch, n_dims)).float()

# print(normal_idx)
# print(samples.shape)
print(samples)

# samples[:, normal_idx] = tmp
samples.scatter_(1, normal_idx[:, None], tmp)

# print("direction:")
# print(direction)
# print("normal_idx:")
# print(normal_idx)
print("samples")
print(samples)

plt.scatter(samples[:, 0], samples[:, 1])
plt.savefig("./example.png")