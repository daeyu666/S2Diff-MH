"""Descriptive paired validation; formal std uses independent geometry-seed means."""
from statistics import mean, stdev
import numpy as np

METRICS = ("full_PSNR", "full_SAM", "full_ERGAS", "E_H", "E_M")
LOWER = {"full_SAM", "full_ERGAS", "E_H", "E_M"}


def correlation(x, y):
    if len(x) < 3:
        return None
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    a, b = a-a.mean(), b-b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a.dot(b)/denom, -1, 1)) if denom > 1e-15 else None


def validation_reports(rows):
    """Cases -> patches -> seed means; no pooled-case std substituted for seed std."""
    from scipy.stats import rankdata
    statistics, seeds, improvements, correlations = [], [], [], []
    for condition, output in dict.fromkeys((r["condition"],r["output"]) for r in rows):
        group = [r for r in rows if (r["condition"],r["output"]) == (condition,output)]
        seed_ids = sorted({r["geometry_seed"] for r in group if r.get("geometry_seed") is not None})
        per_seed = []
        for seed in seed_ids:
            selected = [r for r in group if r["geometry_seed"] == seed]
            patches = sorted({r["sample"] for r in selected})
            values = {metric: mean(mean(r[metric] for r in selected if r["sample"] == patch) for patch in patches)
                      for metric in METRICS}
            per_seed.append(values)
            seeds.append({"condition":condition,"output":output,"geometry_seed":seed,**values})
        for metric in METRICS:
            statistics.append({"condition":condition,"output":output,"metric":metric,
                "mean":mean(v[metric] for v in per_seed) if per_seed else mean(r[metric] for r in group),
                "std_between_seed_means_ddof1":stdev(v[metric] for v in per_seed) if len(per_seed)>1 else None,
                "std_between_observations_ddof1":stdev(r[metric] for r in group) if len(group)>1 else None,
                "n_geometry_seeds":len(seed_ids),"n_observations":len(group)})
    for condition in dict.fromkeys(r["condition"] for r in rows):
        selected = [r for r in rows if r["condition"]==condition]
        identities = list(dict.fromkeys((r["sample"],r["case"]) for r in selected))
        indexed = {(r["sample"],r["case"],r["output"]):r for r in selected}
        for reference in ("state_x0","last_prediction"):
            amplitude = [indexed[s,c,"corrected_x0"]["estimated_terminal_residual_L2"] for s,c in identities]
            for metric in METRICS:
                gains = [(indexed[s,c,reference][metric]-indexed[s,c,"corrected_x0"][metric]
                          if metric in LOWER else indexed[s,c,"corrected_x0"][metric]-indexed[s,c,reference][metric])
                         for s,c in identities]
                improvements.append({"condition":condition,"reference":reference,"metric":metric,
                    "improved_cases":sum(g>0 for g in gains),"tied_cases":sum(g==0 for g in gains),
                    "total_cases":len(gains),"improved_fraction":mean(g>0 for g in gains),
                    "mean_gain":mean(gains),"median_gain":float(np.median(gains))})
                if condition=="warp":
                    correlations.append({"condition":condition,"reference":reference,"metric_gain":metric,
                        "amplitude":"estimated_terminal_residual_L2","n_observations":len(gains),
                        "pearson":correlation(amplitude,gains),
                        "spearman":correlation(rankdata(amplitude),rankdata(gains)),
                        "interpretation":"pooled descriptive association; no causal/significance claim"})
    return {"validation_statistics.csv":statistics,"validation_per_seed.csv":seeds,
            "improvement_rates.csv":improvements,"gain_correlations.csv":correlations}

