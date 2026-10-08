"""SRF-projected, MSI-domain modified QNR for real Augsburg-2 Wald x3.

Metric equations and default evaluation policy are preserved from
comparison_experiments/comparison/UAFL/augsburg2_wald_qnr.py.
This file makes the same metric independently available in S2Diff-MH.

This is an explicitly defined HSI/MSI adaptation, NOT the classical
single-PAN QNR or a full-242-band spectral fidelity score.

Let F be fused 10m HSI (242), H the OBSERVED 30m HSI (242),
M the OBSERVED 10m real S2 MSI (4), R the fixed 4x242 SRF,
and M↓3 area-integrated to 30m. M has the same radiometry correction
as used as input to UAFL (no geometric image warp).

A=R(F), B=R(H).
Q(a,b) is masked, non-overlapping-window UIQI.
Dlambda=mean_{i<j}|Q(A_i,A_j)-Q(B_i,B_j)|, i,j in 4 MSI bands.
Ds=mean_{i,j}|Q(A_i,M_j)-Q(B_i,M↓3_j)|, i,j in 4 MSI bands.
QNR=max(0,1-Dlambda)*max(0,1-Ds).

Report these as MSI_projected_QNR, MSI_projected_Dlambda,
MSI_projected_Ds, not as full-spectrum mQNR or classical PAN-QNR.
The metric is affected by residual true sensor registration and
radiometric discrepancies. No 10m HSI GT is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _masked_uiqi(a, b, mask, window, min_valid_fraction=0.8):
    """Average local UIQI, weighting each window by number of valid pixels."""
    if a.shape != b.shape or a.shape != mask.shape or a.ndim != 2:
        raise ValueError("UIQI spatial arrays must share 2D shape")
    h,w=a.shape
    value_sum,weight_sum=0.,0.
    for y in range(0,h,window):
        for x in range(0,w,window):
            yh=min(y+window,h)
            xw=min(x+window,w)
            tile_mask=mask[y:yh,x:xw]
            total=tile_mask.size
            valid=tile_mask & np.isfinite(a[y:yh,x:xw]) & np.isfinite(b[y:yh,x:xw])
            n=int(valid.sum())
            if n<8 or n/total<float(min_valid_fraction):
                continue
            aa=np.asarray(a[y:yh,x:xw][valid],dtype=np.float64)
            bb=np.asarray(b[y:yh,x:xw][valid],dtype=np.float64)
            ma=float(aa.mean());mb=float(bb.mean())
            va=float(((aa-ma)**2).mean());vb=float(((bb-mb)**2).mean())
            cov=float(((aa-ma)*(bb-mb)).mean())
            luma=2*ma*mb/(ma*ma+mb*mb+1e-12)
            if va+vb<1e-12:
                contrast=1.0
            else:
                contrast=2*cov/(va+vb)
            q=float(np.clip(luma*contrast,-1.,1.))
            value_sum+=q*n
            weight_sum+=n
    if weight_sum==0:
        raise ValueError("No valid local UIQI windows; check masks/window size")
    return value_sum/weight_sum


def projected_qnr(fused, observed_lr_hsi, observed_hr_msi, valid_hr,
                  srf_weights, gains=None, biases=None, window_hr=48,
                  min_valid_fraction=0.8):
    """Compute reproducible fixed-SRF MSI-domain QNR, Dlambda and Ds."""
    fused=np.asarray(fused)
    lr=np.asarray(observed_lr_hsi)
    msi=np.asarray(observed_hr_msi)
    mask=np.asarray(valid_hr,dtype=bool)
    r=np.asarray(srf_weights,dtype=np.float64)
    h,w,c=fused.shape
    if c!=242 or r.shape!=(4,242) or msi.shape!=(h,w,4) or mask.shape!=(h,w):
        raise ValueError("Wald arrays must be fused HxWx242, MSI HxWx4, mask HxW, SRF 4x242")
    if h%3 or w%3 or lr.shape!=(h//3,w//3,242):
        raise ValueError("Wald LR-HSI must have exactly one-third spatial resolution")
    if window_hr%3 or window_hr<24:
        raise ValueError("window_hr must be >=24 and divisible by 3")
    if not np.isfinite(r).all() or (r<0).any() or not np.allclose(r.sum(1),1.,atol=1e-4):
        raise ValueError("Invalid fixed Sentinel-2 response matrix")
    gains=np.ones(4,dtype=np.float64) if gains is None else np.asarray(gains,dtype=np.float64)
    biases=np.zeros(4,dtype=np.float64) if biases is None else np.asarray(biases,dtype=np.float64)
    if gains.shape!=(4,) or biases.shape!=(4,):
        raise ValueError("4-channel radiometry required")

    # Eval on reflectance scale, with the SAME input radiometry, not a new fit.
    corrected_msi=(msi.astype(np.float32)*gains[None,None,:]+biases[None,None,:]).astype(np.float32)
    high_proj=(fused.astype(np.float32)@r.T.astype(np.float32)).astype(np.float32)
    low_proj=(lr.astype(np.float32)@r.T.astype(np.float32)).astype(np.float32)
    msi_low=corrected_msi.reshape(h//3,3,w//3,3,4).mean((1,3))
    mask_low=mask.reshape(h//3,3,w//3,3).all((1,3))

    hiwin=window_hr
    lowin=window_hr//3
    def highq(a,b):
        return _masked_uiqi(a,b,mask,hiwin,min_valid_fraction)
    def lowq(a,b):
        return _masked_uiqi(a,b,mask_low,lowin,min_valid_fraction)
    spectral=[]
    for i in range(4):
        for j in range(i+1,4):
            spectral.append(abs(
                highq(high_proj[:,:,i],high_proj[:,:,j])
                -lowq(low_proj[:,:,i],low_proj[:,:,j])
            ))
    spatial=[]
    for i in range(4):
        for j in range(4):
            spatial.append(abs(
                highq(high_proj[:,:,i],corrected_msi[:,:,j])
                -lowq(low_proj[:,:,i],msi_low[:,:,j])
            ))
    dl=float(np.mean(spectral))
    ds=float(np.mean(spatial))
    return {
        "QNR":float(max(0.,1-dl)*max(0.,1-ds)),
        "Dlambda":dl,
        "Ds":ds,
        "index_name":"MSI-projected modified QNR (fixed measured SRF)",
        "spectral_pair_count":len(spectral),
        "spatial_pair_count":len(spatial),
        "high_window":int(window_hr),
        "low_window":int(lowin),
        "window_min_valid_fraction":float(min_valid_fraction),
        "high_valid_pixels":int(mask.sum()),
        "low_valid_pixels":int(mask_low.sum()),
        "spectral_domain":"4 Sentinel-2-projected bands, not all 242 HSI bands",
        "spatial_reference":"real Sentinel-2 B2/B3/B4/B8, raw geometry with train-only radiometry",
        "source_HSI":"observed 30m HSI",
        "source_MSI":"actual 10m Sentinel-2",
        "full_HR_HSI_ground_truth_used":False,
    }


def evaluate_cache(wald_root, fused, radiometry_json,
                   window_hr=48, min_valid_fraction=0.8):
    root=Path(wald_root)
    with (root/"full"/"meta.json").open() as f:
        meta=json.load(f)
    if meta.get("region")!="sub_area_2":
        raise ValueError("Original-scale Wald QNR restricted to sub_area_2")
    # Strictly require the same Wald cache provenance and the exact
    # pre-existing calibration; do not fit gains/biases on the full test image.
    from train_augsburg_real_cdrdi import _wald_metadata
    if not _wald_metadata(str(root)):
        raise ValueError("Expected strict Augsburg-2 Wald provenance in train/validation/test")
    with Path(radiometry_json).open(encoding="utf-8") as f:
        radiometry = json.load(f)
    gain = np.asarray(radiometry["gain"], dtype=np.float64)
    bias = np.asarray(radiometry["bias"], dtype=np.float64)
    return projected_qnr(
        np.load(fused,mmap_mode="r"),
        np.load(root/"full"/"lr_hsi.npy",mmap_mode="r"),
        np.load(root/"full"/"hr_msi.npy",mmap_mode="r"),
        np.load(root/"full"/"valid_mask.npy",mmap_mode="r"),
        np.load(root/"srf_weights.npy"),
        gain,bias,window_hr,min_valid_fraction,
    )


def main():
    p=argparse.ArgumentParser(description="Wald x3 original-scale MSI-projected modified QNR")
    p.add_argument("--wald_root",default="./data/augsburg2_wald")
    p.add_argument("--fused",required=True)
    p.add_argument("--radiometry_json",default="./data/calibration/Augsburg2_Wald_radiometry.json")
    p.add_argument("--window_hr",type=int,default=48)
    p.add_argument("--min_valid_fraction",type=float,default=.8)
    p.add_argument("--output_json",default="")
    args=p.parse_args()
    m=evaluate_cache(args.wald_root,args.fused,args.radiometry_json,
                     args.window_hr,args.min_valid_fraction)
    print(f"S2DIFF_MH_WALD_ORIGINAL_MSI_QNR QNR={m['QNR']:.6f} Dlambda={m['Dlambda']:.6f} Ds={m['Ds']:.6f}")
    if args.output_json:
        path=Path(args.output_json)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(m,indent=2,ensure_ascii=False))
        print(f"QNR_JSON={path}")


if __name__=="__main__":
    main()
