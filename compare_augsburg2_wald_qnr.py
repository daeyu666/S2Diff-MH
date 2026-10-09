"""Compare strictly equivalent UAFL and S2Diff-MH original-scale Wald QNR JSON.

Compares full-resolution HSI-MSI QNR with *all* 242 LR-HSI spectral
bands and observed 4-band real HR-MSI spatial reference. No PAN/proxy. Metric-definition and valid-domain metadata
MUST match before numerical comparison.
"""
import argparse
import json
import math


COMPARE_FIELDS = (
    "index_name", "qnr_equations",
    "spectral_pair_count", "spatial_pair_count",
    "spatial_support_counts", "srf_support_fraction_of_peak",
    "high_window", "low_window", "window_min_valid_fraction",
    "high_valid_pixels", "low_valid_pixels",
    "spectral_reference", "spatial_reference",
    "source_HSI", "source_MSI", "pan_used", "srf_projection_used",
    "evaluation_area", "test_bbox_30m", "test_bbox_10m", "split_protocol_id",
    "full_HR_HSI_ground_truth_used",
)


def compare_qnr(a, b):
    for key in COMPARE_FIELDS:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(
                f"Different UAFL / S2Diff-MH metric conditions for {key}: "
                f"{a.get(key)!r} versus {b.get(key)!r}"
            )
    if a.get("full_HR_HSI_ground_truth_used") is not False:
        raise ValueError("Not a valid no-10m-HSI-reference QNR comparison")
    if a.get("qnr_equations") != "QNR-HSI-MSI-spectral-all242-spatial-SRF-covered-p=q=alpha=beta=1":
        raise ValueError("Requires HSI-MSI QNR, not old MSI-projected or PAN scores")
    if a.get("spectral_pair_count") != 29161:
        raise ValueError("Spectral distortion must cover all 242 HSI bands")
    if a.get("pan_used") is not False or a.get("srf_projection_used") is not False:
        raise ValueError("Original HSI/MSI observations must be used without PAN/projection")
    for name, row in (("UAFL", a), ("S2Diff-MH", b)):
        if any(k not in row or not math.isfinite(float(row[k]))
               for k in ("QNR", "Dlambda", "Ds")):
            raise ValueError(f"{name}: invalid QNR/Dlambda/Ds")
    return {key: float(b[key]) - float(a[key]) for key in ("QNR", "Dlambda", "Ds")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--center_holdout", action="store_true",
                   help="Compare center test QNR from each model's own results folder")
    p.add_argument("--uafl_json", default=None)
    p.add_argument("--s2diff_json", default=None)
    args = p.parse_args()
    if args.center_holdout:
        args.uafl_json = args.uafl_json or (
            "../comparison_experiments/comparison/UAFL/outputs/"
            "augsburg2_wald_center_holdout/UAFL_Wald_heldout_QNR.json"
        )
        args.s2diff_json = args.s2diff_json or (
            "./outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_heldout_QNR.json"
        )
    elif args.uafl_json is None:
        p.error("Pass --center_holdout or supply --uafl_json for legacy full-region QNR")
    if args.s2diff_json is None:
        args.s2diff_json = "./outputs/augsburg2_wald/Augsburg2_Wald_full_QNR.json"
    with open(args.uafl_json, encoding="utf-8") as f:
        uafl = json.load(f)
    with open(args.s2diff_json, encoding="utf-8") as f:
        ours = json.load(f)
    diff = compare_qnr(uafl, ours)
    print(f"METRIC={uafl['index_name']} PAN_USED={uafl['pan_used']}")
    print(
        f"UAFL QNR={uafl['QNR']:.6f} "
        f"Dlambda={uafl['Dlambda']:.6f} Ds={uafl['Ds']:.6f}"
    )
    print(
        f"S2DIFF_MH QNR={ours['QNR']:.6f} "
        f"Dlambda={ours['Dlambda']:.6f} Ds={ours['Ds']:.6f}"
    )
    print(
        f"S2DIFF_MINUS_UAFL QNR={diff['QNR']:+.6f} "
        f"Dlambda={diff['Dlambda']:+.6f} Ds={diff['Ds']:+.6f}"
    )


if __name__ == "__main__":
    main()
