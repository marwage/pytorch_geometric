import torch


def dropout(x, p, training, device=None):
    if training:
        mask = torch.bernoulli(torch.full_like(x, p, device=device))
        output = torch.mul(x, mask)
        mask = mask.cpu()
        return output
    else:
        return x
