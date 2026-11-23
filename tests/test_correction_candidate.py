"""Candidate output routing, seed protocol, paired statistics and refiner training."""
import json
import sys
from pathlib import Path
from statistics import stdev

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import diagnose_diffusion_endpoint as entry
import train_innovation3_gigi_cdrdi as pipeline
from innovation1 import reconstruct_from_terminal_lr
from diagnose_terminal_residual_correction import correct_endpoint, main as replay_main
from correction_validation_statistics import validation_reports
from diffusion_output_mode import check_checkpoint_mode


@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class Predictor(torch.nn.Module):
    def forward(self,state,t):
        return state*.85 + .05 + t[:,None,None,None]*.0001


def test_online_matches_offline_and_legacy_default(tmp_path):
    args=entry.parse_args(["--device","cpu","--diffusion_steps","4"])
    process=pipeline.build_progressive_process(pipeline._config(args))
    gt=torch.rand(1,4,64,64)
    observed=process.terminal_observation(gt)
    pair=entry.reconstruct_pair(Predictor(),process,observed,target_size=(64,64),hr_msi=None)
    expected,_=correct_endpoint(pair["state_x0"],pair["last_prediction"],pair["terminal_state"],process)
    for mode,target in (("state",pair["state_x0"]),("terminal_correction",expected)):
        result=reconstruct_from_terminal_lr(Predictor(),process,observed,target_size=(64,64),output_mode=mode)
        torch.testing.assert_close(result,target,atol=0,rtol=0)
    with pytest.raises(ValueError):
        reconstruct_from_terminal_lr(Predictor(),process,observed,target_size=(64,64),output_mode="typo")


def test_native_x3_process_has_corrected_oracle_endpoint():
    from augsburg_real_process import build_augsburg_real_process
    process=build_augsburg_real_process(effective_sigma=1.2,diffusion_steps=12)
    gt=torch.rand(1,16,48,48)
    observed=process.terminal_observation(gt)+.003
    class Oracle(torch.nn.Module):
        def forward(self,state,t):
            return gt
    original=reconstruct_from_terminal_lr(Oracle(),process,observed,target_size=(48,48))
    corrected=reconstruct_from_terminal_lr(Oracle(),process,observed,target_size=(48,48),output_mode="terminal_correction")
    assert float((original-gt).square().mean().sqrt())>1e-5
    torch.testing.assert_close(corrected,gt,atol=2e-5,rtol=0)


def test_main_table_cases_match_shared_sampler_and_replay_statistics(tmp_path):
    from fixed_deformation_protocol import warp_hsi
    from multiseed_deformation_protocol import sample_geometry, tensor_sha256, private_seed
    from cdrdi_geometry import spectral_project
    args=entry.parse_args(["--protocol","main_table","--device","cpu","--diffusion_steps","4",
        "--geometry_seeds","10001","10002","--cases","2","--error_decomposition","--output_dir",str(tmp_path/"source")])
    process=pipeline.build_progressive_process(pipeline._config(args))
    gt=torch.rand(1,4,128,128)
    srf=torch.ones(2,4)/4
    msi=spectral_project(gt,srf)
    def estimate(observed,msi):
        return torch.zeros(1,3),torch.zeros(1,2,128,128)
    with torch.no_grad():
        entry.run_diagnostic(args,[{"gt":gt[0],"hr_msi":msi[0]}],process,Predictor(),estimate,srf,torch.device("cpu"))
    root=Path(args.output_dir)
    traces=json.loads((root/"traces.json").read_text())
    warp=[r for r in traces if r["condition"]=="warp"]
    assert len(warp)==4
    for r in warp:
        phi,_=sample_geometry(128,128,geometry_seed=r["geometry_seed"],sample_index=0,case_index=r["case_within_seed"]-1)
        observed=process.operator.degrade(warp_hsi(gt,phi))
        assert r["observed_sha256"]==tensor_sha256(observed)
        assert r["reconstruction_seed"]==private_seed("reconstruction",args.eval_seed,0,r["case_within_seed"]-1)
    entry.write_json(root/"manifest.json",{"complete":True,"protocol":"paired-diffusion-endpoint-v1",
        "args":vars(args),"full_test_samples":1,"srf_sha256":entry.tensor_hash(srf),
        "error_decomposition":{"enabled":True}})
    replay_main(["--source_dir",str(root),"--output_dir",str(tmp_path/"replay"),"--device","cpu","--require_decomposition"])
    for name in ("validation_statistics.csv","validation_per_seed.csv","improvement_rates.csv","gain_correlations.csv"):
        assert (tmp_path/"replay"/name).is_file()


def test_seed_std_and_signed_correlations_are_not_case_std():
    rows=[]
    for i,(seed,score,amplitude) in enumerate([(1,1.,1.),(1,3.,2.),(2,7.,3.),(2,9.,4.)]):
        for output,gain in (("state_x0",0.),("last_prediction",.5),("corrected_x0",amplitude)):
            rows.append({"condition":"warp","sample":1,"case":i+1,"geometry_seed":seed,"output":output,
                "full_PSNR":score+gain,"full_SAM":10-score-gain,"full_ERGAS":20-score-gain,
                "E_H":.01+gain*.001,"E_M":.02-gain*.001,"estimated_terminal_residual_L2":amplitude})
    reports=validation_reports(rows)
    row=next(r for r in reports["validation_statistics.csv"] if r["output"]=="state_x0" and r["metric"]=="full_PSNR")
    assert row["std_between_seed_means_ddof1"]==pytest.approx(stdev([2.,8.]))
    assert row["std_between_observations_ddof1"]==pytest.approx(stdev([1.,3.,7.,9.]))
    corr=next(r for r in reports["gain_correlations.csv"] if r["reference"]=="state_x0" and r["metric_gain"]=="full_SAM")
    assert corr["pearson"]==pytest.approx(1.)
    rate=next(r for r in reports["improvement_rates.csv"] if r["reference"]=="state_x0" and r["metric"]=="E_H")
    assert rate["improved_cases"]==0


def test_refiner_mode_guard_and_separate_paths(tmp_path):
    parser=pipeline.build_arg_parser()
    args=parser.parse_args(["--checkpoint_root",str(tmp_path),"--log_root",str(tmp_path),"--save_name","candidate"])
    original=pipeline._checkpoint_path(args)
    args.diffusion_output="terminal_correction"
    candidate=pipeline._checkpoint_path(args)
    assert original!=candidate and candidate.endswith("_terminal_correction.pth")
    check_checkpoint_mode({"extra":{}},"state")
    with pytest.raises(ValueError,match="diffusion_output mismatch"):
        check_checkpoint_mode({"extra":{}},"terminal_correction")
    with pytest.raises(ValueError,match="diffusion_output mismatch"):
        check_checkpoint_mode({"extra":{"diffusion_output":"terminal_correction"}},"state")


def test_corrected_refiner_one_epoch_recomputes_residual_and_freezes_backbones(tmp_path,monkeypatch):
    args=pipeline.build_arg_parser().parse_args(["--device","cpu","--dataset","PaviaU",
        "--geometry_base_channels","8","--diffusion_base_channels","8","--time_dim","32","--spectral_hidden","2",
        "--refine_hidden","8","--heads","1","--geometry_steps","1","--diffusion_steps","4","--eval_cases","1",
        "--epochs","1","--eval_interval","1","--save_interval","1","--from_scratch",
        "--diffusion_output","terminal_correction","--checkpoint_root",str(tmp_path/"ckpt"),"--log_root",str(tmp_path/"logs")])
    info={"n_bands":31,"n_msi_bands":3,"srf_weights":np.full((3,31),1/31)}
    device=torch.device("cpu")
    geometry,diffusion=pipeline._build_frozen_models(args,info,device,load_weights=False)
    for role,model in (("geometry",geometry),("diffusion",diffusion)):
        path=tmp_path/(role+".pth")
        torch.save({"model":model.state_dict()},path)
        setattr(args,role+"_checkpoint",str(path))
    gt=torch.rand(31,64,64)
    msi=torch.einsum("mc,chw->mhw",torch.as_tensor(info["srf_weights"],dtype=torch.float32),gt)
    loader=torch.utils.data.DataLoader([{"gt":gt,"hr_msi":msi}],batch_size=1)
    monkeypatch.setattr(pipeline,"build_train_val_test_loaders",lambda cfg:(loader,loader,loader,info))
    frozen=[]
    build=pipeline._build_frozen_models
    def build_frozen(*a,**kw):
        models=build(*a,**kw)
        frozen.extend((m,{k:v.clone() for k,v in m.state_dict().items()}) for m in models)
        return models
    monkeypatch.setattr(pipeline,"_build_frozen_models",build_frozen)
    reconstruct=pipeline.reconstruct_from_terminal_lr
    residual=pipeline._physical_residual
    seen=[]
    def reconstruct_record(*a,**kw):
        assert kw["output_mode"]=="terminal_correction"
        result=reconstruct(*a,**kw)
        seen.append(result.clone())
        return result
    def residual_record(process,projector,observed,base):
        torch.testing.assert_close(base,seen[-1],atol=0,rtol=0)
        return residual(process,projector,observed,base)
    monkeypatch.setattr(pipeline,"reconstruct_from_terminal_lr",reconstruct_record)
    monkeypatch.setattr(pipeline,"_physical_residual",residual_record)
    pipeline.train(args)
    assert len(seen)>=3
    for model,before in frozen:
        assert all(not p.requires_grad for p in model.parameters())
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    state=torch.load(pipeline._checkpoint_path(args),map_location="cpu",weights_only=False)
    assert state["extra"]["diffusion_output"]=="terminal_correction"
    refiner=pipeline._build_refiner(args,info,device)
    pipeline._load_training_checkpoint(refiner,pipeline._checkpoint_path(args),device)
    args.diffusion_output="state"
    refiner=pipeline._build_refiner(args,info,device)
    with pytest.raises(ValueError,match="diffusion_output mismatch"):
        pipeline._load_training_checkpoint(refiner,str(next((tmp_path/"ckpt").glob("*terminal_correction.pth"))),device)


def test_real_native_parser_exposes_same_mode(monkeypatch):
    import infer_augsburg2_wald as base
    import infer_augsburg2_wald_gigi as full
    monkeypatch.setattr(sys,"argv",["infer","--center_holdout","--diffusion_output","terminal_correction"])
    args=base.parse_args()
    assert args.save_root.endswith("/terminal_correction")
    monkeypatch.setattr(sys,"argv",["infer","--center_holdout","--refiner_checkpoint","candidate.pth",
                                  "--diffusion_output","terminal_correction"])
    args=full.parse_args()
    assert args.save_root.endswith("/terminal_correction")

