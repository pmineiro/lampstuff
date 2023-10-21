def SimpleRegretHypercubeSampler(fhat, gamma):
    from math import sqrt
    import torch

    A = fhat.shape[1]
    assert A >= 1

    if A == 1:
        exploit = torch.zeros_like(fhat).long()
        explore = torch.empty(fhat.shape[0], 0, device=fhat.device, dtype=torch.long)
    else:
        fhatahats, ahats = fhat.topk(k=2, dim=1)
        gamma *= sqrt(A)

        fhat -= fhatahats[:,[0]]
        maxterm = torch.clamp(2 + gamma * fhatahats[:,[1]], min=0, max=None)
        z = 1 / torch.clamp(-1 - gamma * fhat + maxterm, min=1)
        z[range(z.shape[0]), ahats[:,0]] = 0
        exploit = ahats[:,0]
        explore = torch.bernoulli(z)

    return exploit, explore
