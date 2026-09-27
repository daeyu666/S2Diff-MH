import pytest
import torch
from evaluation_protocol import evaluate_all_samples


def test_evaluation_uses_every_sample_and_retains_worst_case():
    seen = []

    @evaluate_all_samples
    def evaluate(model, test_loader, *, offset):
        v = next(iter(test_loader))["gt"].item()
        seen.append(v)
        return {"metric": {"PSNR": v + offset}, "min_pred_jacobian": v}, [v, 2*v]

    batches = [{"gt": torch.tensor([float(v)])} for v in [1, 3, 8]]
    result = evaluate(None, batches, offset=10)
    assert seen == [1, 3, 8]
    assert result == ({"metric": {"PSNR": 14}, "min_pred_jacobian": 1}, [4, 8])
    with pytest.raises(ValueError, match="empty"):
        evaluate(None, [], offset=10)
