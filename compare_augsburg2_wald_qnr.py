"""Compare strictly equivalent UAFL and S2Diff-MH original-scale Wald QNR JSON.

The number follows the classical QNR equations with a defined PAN
source. Augsburg defaults to a *synthetic MSI-derived PAN*, not real PAN.
Do not call it a 242-band spectral fidelity index. Metric-definition and valid-domain metadata
MUST match before numerical comparison.
"""
import argparse
import json
import math


COMPARE_FIELDS = (
    "index_name", "qnr_equations", "pan_origin", "is_genuine_pan",
    "low_resolution_multispectral_reference",
    "spectral_pair_count", "spatial_pair_count",
    "high_window", "low_window", "window_min_valid_fraction",
    "high_valid_pixels", "low_valid_pixels",
    "spectral_domain", "source_HSI", "source_MSI",
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
    if a.get("qnr_equations") != "Alparone-2008-classical-formula-p=q=alpha=beta=1":
        raise ValueError("Only standard-form QNR reports can be compared")
    if a.get("spectral_pair_count") != 6 or a.get("spatial_pair_count") != 4:
        raise ValueError("Classical QNR requires 6 spectral and 4 PAN-spatial comparisons")
    for name, row in (("UAFL", a), ("S2Diff-MH", b)):
        if any(k not in row or not math.isfinite(float(row[k]))
               for k in ("QNR", "Dlambda", "Ds")):
            raise ValueError(f"{name}: invalid QNR/Dlambda/Ds")
    return {key: float(b[key]) - float(a[key]) for key in ("QNR", "Dlambda", "Ds")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uafl_json", required=True)
    p.add_argument("--s2diff_json", default="./outputs/augsburg2_wald/Augsburg2_Wald_full_QNR.json")
    args = p.parse_args()
    with open(args.uafl_json, encoding="utf-8") as f:
        uafl = json.load(f)
    with open(args.s2diff_json, encoding="utf-8") as f:
        ours = json.load(f)
    diff = compare_qnr(uafl, ours)
    print(f"METRIC={uafl['index_name']} GENUINE_PAN={uafl['is_genuine_pan']}")
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
