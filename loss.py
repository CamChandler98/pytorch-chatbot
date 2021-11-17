import torch
from processed_data import device


def maskNLLLoss(decoder_out, target, mask):
    nTotal  = mask.sum()

    target = target.view(-1,1)

    gathered_tensor = torch.gather(decoder_out,1  ,target)

    cross_entropy = -torch.log(gathered_tensor)

    loss = cross_entropy.masked_select(mask)

    loss = loss.mean()
    loss = loss.to(device)

    return loss, nTotal.item()
