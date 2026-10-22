from __future__ import annotations
from collections import OrderedDict
import torch


def fedavg(state_dicts, weights, keep_on_device: bool = True, client_ids=None):
    if not state_dicts or len(state_dicts) != len(weights):
        raise ValueError("FedAvg requires nonempty matching states and weights")
    if client_ids is not None:
        if len(client_ids) != len(state_dicts) or len(set(client_ids)) != len(client_ids):
            raise ValueError("FedAvg client IDs must be unique and match states")
        order = sorted(range(len(client_ids)), key=lambda i: client_ids[i])
        state_dicts = [state_dicts[i] for i in order]
        weights = [weights[i] for i in order]
    new_sd = OrderedDict()
    total = float(sum(weights))
    if total == 0.0:
        scales = [1.0 / max(1, len(weights))] * len(weights)
    else:
        scales = [float(w) / total for w in weights]
    with torch.no_grad():
        for k in state_dicts[0].keys():
            v0 = state_dicts[0][k]
            if isinstance(v0, torch.Tensor):
                if not v0.is_floating_point():
                    new_sd[k] = v0.detach().clone()
                    continue
                if keep_on_device:
                    acc = v0.detach().clone().mul_(scales[0])
                    for i in range(1, len(state_dicts)):
                        vi = state_dicts[i][k]
                        if vi.device != acc.device:
                            vi = vi.to(acc.device)
                        acc.add_(vi, alpha=scales[i])
                else:
                    acc = v0.detach().clone().mul_(scales[0])
                    for i in range(1, len(state_dicts)):
                        acc.add_(state_dicts[i][k], alpha=scales[i])
                new_sd[k] = acc
            else:
                new_sd[k] = v0
    return new_sd
