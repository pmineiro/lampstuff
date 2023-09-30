def SimpleRegretGreedyIGW(fhat, gamma):
    from math import sqrt
    import torch

    fhatahats, ahats = fhat.topk(k=2, dim=1)
    A = fhat.shape[1]
    gamma *= sqrt(A)

    z = 1 / (A + gamma * torch.clamp(fhatahats[:,[1]] - fhat, min=0, max=None))
    sumz = z.sum(dim=1)
    z[range(z.shape[0]), ahats[:,0]] += torch.clamp(1 - sumz, min=0, max=None)
    return torch.multinomial(z, num_samples=1).squeeze(1), ahats[:,0]

def SimpleRegretGreedyDoubleSampler(fhat, gamma):
    from math import sqrt
    import torch

    fhatahats, ahats = fhat.topk(k=2, dim=1)
    A = fhat.shape[1]
    gamma *= sqrt(A)

    z = 1 / ((A - 1) + gamma * torch.clamp(fhatahats[:,[1]] - fhat, min=0, max=None))
    z[range(z.shape[0]), ahats[:,0]] = 0
    sumz = z.sum(dim=1)
    z[range(z.shape[0]), ahats[:,1]] += torch.clamp(1 - sumz, min=0, max=None)
    return torch.multinomial(z, num_samples=1).squeeze(1), ahats[:,0]

def SimpleRegretHypercubeSampler(fhat, gamma):
    from math import sqrt
    import torch

    fhatahats, ahats = fhat.topk(k=2, dim=1)
    A = fhat.shape[1]
    gamma *= sqrt(A)

    fhat -= fhatahats[:,[0]]
    maxterm = torch.clamp(2 + gamma * fhatahats[:,[1]], min=0, max=None)
    z = 1 / (-1 - gamma * fhat + maxterm)
    z[range(z.shape[0]), ahats[:,0]] = 0
    return torch.bernoulli(z), ahats[:,0]
