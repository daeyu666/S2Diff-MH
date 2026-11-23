"""Read-only diagnostics of a fixed linear lifted acquisition trajectory.

The collector accumulates L_(t-1)(prediction-GT)-L_t(prediction-GT) directly
at each actual inference step. C_pred is NOT inferred from the final error.
No clipped/normalized images are used. GT never corrects the reverse state.
"""
from __future__ import annotations

from statistics import mean

import torch


def _values(value, mask):
    return value.double().masked_select(mask.expand_as(value))


def measure_terms(delta, correction, actual_error, mask):
    """Vector identities on one common support; zero-vector cosine is undefined."""
    d, c, a = (_values(v, mask) for v in (delta, correction, actual_error))
    if not d.numel() or not all(torch.isfinite(v).all() for v in (d, c, a)):
        raise ValueError("Empty or nonfinite error-decomposition support")
    dn, cn = d.norm(), c.norm()
    total = d + c
    cross = 2 * (d * c).mean()
    c_mse = c.square().mean()
    delta_mse = d.square().mean()
    total_mse = total.square().mean()
    actual_mse = a.square().mean()
    change = cross + c_mse
    cosine = float(((d*c).sum() / (dn*cn)).clamp(-1, 1)) if float(dn) > 1e-12 and float(cn) > 1e-12 else None
    # Tiny numerical changes do not count as evidence of net compensation.
    actual_closure_mse_error = float((actual_mse-total_mse).abs())
    tolerance = max(1e-12, float(delta_mse) * 1e-6, actual_closure_mse_error)
    return {"elements": d.numel(),
            "delta_L2": float(dn), "C_pred_L2": float(cn),
            "delta_plus_C_L2": float(total.norm()), "actual_error_L2": float(a.norm()),
            "delta_RMS": float(delta_mse.sqrt()), "C_pred_RMS": float(c_mse.sqrt()),
            "delta_plus_C_RMS": float(total_mse.sqrt()), "actual_error_RMS": float(actual_mse.sqrt()),
            "cos_delta_C": cosine,
            "delta_MSE": float(delta_mse), "C_pred_MSE": float(c_mse),
            "cross_term_2mean_delta_C": float(cross),
            "theoretical_MSE_change_vs_delta": float(change),
            "actual_MSE_change_vs_delta": float(actual_mse-delta_mse),
            "opposing_component": cosine is not None and cosine < 0,
            "compensation_tolerance_MSE": tolerance,
            "net_compensation": (float(delta_mse) > 1e-16 and float(change) < -tolerance
                                 and float(actual_mse-delta_mse) < -tolerance),
            "decomposition_max_abs_error": float((a-total).abs().max()),
            "decomposition_RMS_error": float((a-total).square().mean().sqrt()),
            "energy_identity_abs_error": float((total_mse-delta_mse-change).abs()),
            "actual_vs_decomposed_MSE_abs_error": actual_closure_mse_error}


class PredictionErrorCollector:
    """Independent observer; has no access to a mutable production reverse state."""

    def __init__(self, process, reference, hr_mask, *, save_step_errors=False):
        self.process = process
        self.reference = reference.detach().clone()
        self.masks = {"full": torch.ones_like(hr_mask, dtype=torch.bool), "roi": hr_mask.bool().clone()}
        self.correction = torch.zeros_like(reference, dtype=torch.float64)
        self.steps = []
        self.saved_errors = {}
        self.save_step_errors = save_step_errors
        self.next_t = process.total_steps

    @torch.no_grad()
    def observe(self, t, prediction):
        if t != self.next_t:
            raise ValueError(f"Expected descending step {self.next_t}, received {t}")
        self.next_t -= 1
        error = prediction - self.reference
        if not torch.isfinite(error).all():
            raise FloatingPointError(f"Nonfinite prediction error at t={t}")
        previous = self.process.state_at(error, t-1)
        current = self.process.state_at(error, t)
        increment = previous.double() - current.double()
        self.correction.add_(increment)
        for scope, mask in self.masks.items():
            e, inc, cumulative = (_values(v, mask) for v in (error, increment, self.correction))
            self.steps.append({"scope": scope, "t": t, "e_t_L2": float(e.norm()),
                               "e_t_RMS": float(e.square().mean().sqrt()),
                               "increment_L2": float(inc.norm()), "increment_RMS": float(inc.square().mean().sqrt()),
                               "cumulative_C_L2": float(cumulative.norm()),
                               "cumulative_C_RMS": float(cumulative.square().mean().sqrt())})
        if self.save_step_errors:
            self.saved_errors[f"prediction_error_t{t:02d}"] = error.cpu().clone()

    @torch.no_grad()
    def finish(self, pair):
        if self.next_t != 0:
            raise ValueError("Incomplete reverse trajectory in error collector")
        # Keep the diagnostic delta separate from model inputs and updates.
        delta = pair["terminal_state"].double() - self.process.state_at(
            self.reference, self.process.total_steps).double()
        actual = pair["state_x0"].double() - self.reference.double()
        rows = [{"scope": scope, **measure_terms(delta, self.correction, actual, mask)}
                for scope, mask in self.masks.items()]
        # The decomposition requires L0=I and fixed, image-independent linear L_t.
        # A failed identity is reported as a failed diagnostic, never compensation.
        tolerance = 2e-5 + 2e-5 * float(actual.abs().max())
        if rows[0]["decomposition_max_abs_error"] > tolerance:
            raise AssertionError("Error decomposition failed: check fixed linear L_t and L0=I; "
                                 f"max error={rows[0]['decomposition_max_abs_error']:.6g}")
        return rows, {"C_pred_diagnostic_only": self.correction, **self.saved_errors}


def summarize_decomposition(rows):
    """Summarize each condition/support without treating undefined cosine as zero."""
    out = []
    for condition, scope in dict.fromkeys((r["condition"], r["scope"]) for r in rows):
        group = [r for r in rows if r["condition"] == condition and r["scope"] == scope]
        cosines = [r["cos_delta_C"] for r in group if r["cos_delta_C"] is not None]
        numeric = ("delta_L2", "C_pred_L2", "delta_plus_C_L2", "actual_error_L2",
                   "delta_RMS", "C_pred_RMS", "delta_plus_C_RMS", "actual_error_RMS",
                   "delta_MSE", "C_pred_MSE", "cross_term_2mean_delta_C",
                   "theoretical_MSE_change_vs_delta", "actual_MSE_change_vs_delta")
        out.append({"condition": condition, "scope": scope, "cases": len(group),
                    "opposing_cases": sum(r["opposing_component"] for r in group),
                    "net_compensation_cases": sum(r["net_compensation"] for r in group),
                    "defined_cosine_cases": len(cosines),
                    "mean_cos_delta_C": mean(cosines) if cosines else None,
                    "max_decomposition_max_abs_error": max(r["decomposition_max_abs_error"] for r in group),
                    "max_energy_identity_abs_error": max(r["energy_identity_abs_error"] for r in group),
                    "max_actual_vs_decomposed_MSE_abs_error": max(r["actual_vs_decomposed_MSE_abs_error"] for r in group),
                    **{"mean_" + k: mean(r[k] for r in group) for k in numeric}})
    return out
