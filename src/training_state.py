"""Small training-state helpers shared by experiment engines."""
from __future__ import annotations

import torch


def snapshot_state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return an immutable CPU snapshot of a model state dict.

    ``Tensor.cpu()`` can share storage with an already-CPU model tensor. Cloning
    after detaching prevents later optimizer steps from mutating the saved
    "best" checkpoint snapshot.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
