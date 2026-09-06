import torch

from training_state import snapshot_state_dict_cpu


def test_snapshot_state_dict_cpu_clones_cpu_tensors():
    model = torch.nn.Linear(2, 1)
    snapshot = snapshot_state_dict_cpu(model)
    original_weight = snapshot["weight"].clone()

    with torch.no_grad():
        model.weight.add_(10.0)

    assert torch.equal(snapshot["weight"], original_weight)
    assert snapshot["weight"].device.type == "cpu"
