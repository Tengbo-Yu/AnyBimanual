import numpy as np
import torch
import torch.nn.functional as F
import itertools
import h5py
import pandas as pd
import pickle
import cv2
import random
import string

import torch.distributions as pyd
from einops import rearrange, repeat
import math

def pad(x, max_len, axis=1, const=0, mode='pre'):
    """Pads input sequence with given const along a specified dim

    Inputs:
        x: Sequence to be padded
        max_len: Max padding length
        axis: Axis to pad (Default: 1)
        const: Constant to pad with (Default: 0)
        mode: ['pre', 'post'] Specifies whether to add padding pre or post to the sequence
    """

    if isinstance(x, tuple):
        x = np.array(x)

    pad_size = max_len - x.shape[axis]
    if pad_size <= 0:
        return x

    npad = [(0, 0)] * x.ndim
    if mode == 'pre':
        npad[axis] = (pad_size, 0)
    elif mode == 'post':
        npad[axis] = (0, pad_size)
    else:
        raise NotImplementedError

    if isinstance(x, np.ndarray):
        x_padded = np.pad(x, pad_width=npad, mode='constant', constant_values=const)
    elif isinstance(x, torch.Tensor):
        # pytorch starts padding from final dim so need to reverse chaining order
        npad = tuple(itertools.chain(*reversed(npad)))
        x_padded = F.pad(x, npad, mode='constant', value=const)
    else:
        raise NotImplementedError
    return x_padded

def entropy(codes, options, lang_state_embeds):
    """Calculate entropy of options over each batch

    option_codes: [N, D]
    lang_state_embeds: [B, D]
    """
    with torch.no_grad():
        N, D = codes.shape
        lang_state_embeds = lang_state_embeds.reshape(-1, 1, D)

        embed = codes.t()
        flatten = rearrange(lang_state_embeds, '... d -> (...) d')

        distance = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        # probs = (distance/2).exp() / math.sqrt(2 * math.pi)
        cond_probs = torch.softmax(distance / 2, dim=1)

        # dist = pyd.Independent(pyd.Normal(codes, torch.ones_like(codes)), 1)
        # probs = dist.log_prob(lang_state_embeds).exp()  # get probs as B x N

        # get marginal probabilities
        probs = cond_probs.mean(dim=0)

        entropy = (-torch.log2(probs) * probs).sum()

        # calculate conditional entropy with language
        # sum over options, and then take expectation over language
        cond_entropy = (-torch.log2(cond_probs) * cond_probs).sum(1).mean(0)
        return (entropy, cond_entropy)
