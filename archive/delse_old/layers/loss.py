from __future__ import division
import numpy as np
import torch
from torch.nn import functional as F
from layers.lse import *


def bd_loss(phi, sdt, sigma=0.08, dt_max=30, size_average=True):
    dist = torch.mul(torch.pow(phi - sdt, 2), Dirac(sdt, sigma, dt_max) + Dirac(phi.detach(), sigma, dt_max))
    loss = torch.sum(dist)
    if size_average:
        sz = Dirac(sdt, sigma, dt_max).sum()
        loss = loss / sz
    return loss


def class_balanced_bce_loss(outputs, labels, size_average=False, batch_average=True):
    assert(outputs.size() == labels.size())

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    loss_val = -(torch.mul(labels, torch.log(outputs)) + torch.mul((1.0 - labels), torch.log(1.0 - outputs)))

    loss_pos = torch.sum(torch.mul(labels, loss_val))
    loss_neg = torch.sum(torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= torch.numel(labels)
    elif batch_average:
        final_loss /= labels.size()[0]
    return final_loss


def level_map_loss(output, sdt, alpha=1):
    assert (output.size() == sdt.size())
    mse = lambda x: torch.mean(torch.pow(x, 2))

    sdt_loss = mse(sdt - output)
    return alpha*sdt_loss


def vector_field_loss(vf_pred, vf_gt, sdt=None, _print=False):
    # (n_batch, n_channels, H, W)
    vf_pred = F.normalize(vf_pred, p=2, dim=1)
    vf_gt = F.normalize(vf_gt, p=2, dim=1)
    cos_dist = torch.sum(torch.mul(vf_pred, vf_gt), dim=1)
    angle_error = torch.acos(cos_dist * (1-1e-4))

    angle_loss = torch.mean(torch.pow(angle_error, 2))
    if _print:
        print('[vf_loss] vf_loss: ' + str(angle_loss.item()))

    return angle_loss


def LSE_output_loss(phi_T, gts, sdts, epsilon=-1, dt_max=30):
    pixel_loss = class_balanced_bce_loss(Heaviside(phi_T, epsilon=epsilon), gts, size_average=True)
    # boundary_loss = bd_loss(phi_T, sdts, sigma=2.0/dt_max, dt_max=dt_max)
    loss = 100 * pixel_loss
    return loss

