"""Summarize held-out observed 30m HSI Wald-D2 geometry A/B/C experiments.

Reads metrics written by --stage test. Never treats EnMAP10 as ground truth.
"""
import argparse
import json
import math
import os


def load_results(log_root):
    modes = {"A": "identity", "B": "wald_fixed", "C": "wald_cdrdi"}
    output = {}
    for branch, mode in modes.items():
        path = os.path.join(log_root, f"Augsburg2_Wald_D2_{branch}_test.json")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if (
            obj.get("stage") != "Augsburg2-Wald-D2"
            or obj.get("split") != "test"
            or obj.get("reference") != "observed_30m_HSI_only"
            or obj.get("msi_source") != "real_Sentinel_2_Wald_30m"
            or obj.get("geometry_mode") != mode
        ):
            raise ValueError(f"{branch} result does not have strict Wald provenance")
        metrics = obj.get("metrics", {})
        if any(k not in metrics or not math.isfinite(float(metrics[k]))
               for k in ("ref_psnr", "ref_sam", "phy", "msi")):
            raise ValueError(f"{branch} result lacks finite held-out metrics")
        output[branch] = obj

    baseline = output["A"]
    for name, obj in output.items():
        for key in ("cache_root", "effective_sigma", "diffusion_steps",
                    "seed", "radiometry_json", "split"):
            if obj.get(key) != baseline.get(key):
                raise ValueError(
                    f"A/B/C experimental setting mismatch for {name}: {key}"
                )
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_root", default="./logs/augsburg_real")
    args = p.parse_args()
    results = load_results(args.log_root)
    print("WALD_ABC_REFERENCE=heldout_observed_EnMAP_like_HSI_30m")
    print("WALD_ABC_10m_GT=unavailable")
    print("BRANCH REF_PSNR REF_SAM PHY_L1 MSI_L1")
    for branch in ("A", "B", "C"):
        m = results[branch]["metrics"]
        print(
            f"{branch} {m['ref_psnr']:.6f} {m['ref_sam']:.6f} "
            f"{m['phy']:.8f} {m['msi']:.8f}"
        )
    for x, y in (("B", "A"), ("C", "B"), ("C", "A")):
        a, b = results[x]["metrics"], results[y]["metrics"]
        print(
            f"WALD_DELTA {x}_MINUS_{y} "
            f"PSNR={a['ref_psnr']-b['ref_psnr']:+.6f}dB "
            f"SAM={a['ref_sam']-b['ref_sam']:+.6f}deg "
            f"PHY_L1={a['phy']-b['phy']:+.8f} "
            f"MSI_L1={a['msi']-b['msi']:+.8f}"
        )
    print(
        "WALD_ABC_INTERPRETATION=positive_C_minus_B_PSNR and negative_C_minus_B_SAM "
        "indicate an apparent CDRDI reconstruction benefit on this held-out split; "
        "replicate with more seeds before claiming stability"
    )


if __name__ == "__main__":
    main()
