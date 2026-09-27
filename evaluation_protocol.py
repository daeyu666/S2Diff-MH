"""Apply single-patch diagnostics to every held-out patch/scene."""
import functools
import inspect
from statistics import mean


def _aggregate(rows, key=None):
    first = rows[0]
    if isinstance(first, dict):
        return {k: _aggregate([r[k] for r in rows], k) for k in first}
    if isinstance(first, (tuple, list)):
        return type(first)(_aggregate([r[i] for r in rows]) for i in range(len(first)))
    # A minimum Jacobian is a worst-case diagnostic, not an average.
    return min(rows) if key == "min_pred_jacobian" else mean(rows)


def evaluate_all_samples(evaluator):
    """Macro-average all batches (evaluation loaders must use batch_size=1).

    Existing per-sample seeded geometry cases are preserved, so all patches see
    the same deterministic case schedule. Single-patch outputs stay unchanged.
    """
    signature = inspect.signature(evaluator)

    @functools.wraps(evaluator)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        loader = bound.arguments["test_loader"]
        rows = []
        for batch in loader:
            if batch["gt"].shape[0] != 1:
                raise ValueError("Evaluation requires batch_size=1 for macro averaging")
            bound.arguments["test_loader"] = [batch]
            rows.append(evaluator(*bound.args, **bound.kwargs))
        if not rows:
            raise ValueError("Cannot evaluate an empty split")
        return rows[0] if len(rows) == 1 else _aggregate(rows)

    return wrapped
