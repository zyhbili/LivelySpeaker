import torch
import torch.nn.functional as F


def compute_rcxyz_loss(batch, use_txt_output=False):
    x = batch["x"]
    output = batch["output_xyz"]
    if use_txt_output:
        output = batch["txt_output_xyz"]

    loss = F.mse_loss(x, output, reduction='mean')
    return loss



def compute_velxyz_loss(batch, use_txt_output=False):
    x = batch["x"]
    output = batch["output_xyz"]
    if use_txt_output:
        output = batch["txt_output_xyz"]
    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    loss = F.mse_loss(gtvel, outputvel, reduction='mean')
    return loss







