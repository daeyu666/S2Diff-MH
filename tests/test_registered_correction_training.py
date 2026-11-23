"""Registered-only correction training and configuration inheritance."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train_innovation3_gigi as registered
from models import RawMSIDirectPredictor
from innovation1 import reconstruct_from_terminal_lr


@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_recorded_configuration_inherits_budget_but_not_resume_or_output_names(tmp_path):
    path=tmp_path/"old.pth"
    cfg=vars(registered.parse_args(["--epochs","137","--lr",".00013","--batch_size","3","--seed","17",
                                  "--save_name","old","--resume","old_last.pth"]))
    torch.save({"extra":cfg},path)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    args=registered.parse_args(["--config_from_checkpoint",str(path),"--stage","test",
                              "--diffusion_output","terminal_correction","--save_name","new","--lr",".0002"])
    assert (args.epochs,args.batch_size,args.seed)==(137,3,17)
    assert args.lr==.0002 and args.stage=="test" and args.resume==""
    assert args.save_name=="new" and args.diffusion_output=="terminal_correction"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    torch.save({"extra":{"dataset":"PaviaU","geometry_checkpoint":"mixed.pth"}},path)
    with pytest.raises(SystemExit):
        registered.parse_args(["--config_from_checkpoint",str(path)])


def test_registered_corrected_training_validation_and_test(tmp_path,monkeypatch):
    source=tmp_path/"old_registered.pth"
    baseline_path=tmp_path/"registered_diffusion.pth"
    original=registered.parse_args(["--device","cpu","--epochs","1","--eval_interval","1","--save_interval","1",
        "--batch_size","1","--base_channels","8","--time_dim","32","--spectral_hidden","2",
        "--refine_hidden","8","--heads","1","--diffusion_steps","4",
        "--baseline_checkpoint_type","standard","--baseline_checkpoint",str(baseline_path)])
    torch.save({"extra":vars(original)},source)
    args=registered.parse_args(["--config_from_checkpoint",str(source),"--diffusion_output","terminal_correction",
        "--save_name","registered_new","--checkpoint_root",str(tmp_path/"out"),"--log_root",str(tmp_path/"logs")])
    info={"n_bands":31,"n_msi_bands":3,"srf_weights":np.full((3,31),1/31)}
    baseline=RawMSIDirectPredictor(n_bands=31,n_msi_bands=3,total_steps=4,base_channels=8,
        time_dim=32,spectral_hidden=2,dropout=0.,residual_prediction=True)
    torch.save({"model":baseline.state_dict()},baseline_path)
    gt=torch.rand(31,64,64)
    msi=torch.einsum("mc,chw->mhw",torch.as_tensor(info["srf_weights"],dtype=torch.float32),gt)
    loader=torch.utils.data.DataLoader([{"gt":gt,"hr_msi":msi}],batch_size=1)
    monkeypatch.setattr(registered,"build_train_val_test_loaders",lambda cfg:(loader,loader,loader,info))
    monkeypatch.setattr(registered,"build_loaders",lambda cfg:(loader,loader,info))
    records=[]
    load=registered._build_baseline
    def load_frozen(*a,**kw):
        model=load(*a,**kw)
        records.append((model,{k:v.clone() for k,v in model.state_dict().items()}))
        return model
    monkeypatch.setattr(registered,"_build_baseline",load_frozen)
    refiner_initial=[]
    build=registered._build_refiner
    def build_record(*a,**kw):
        model=build(*a,**kw)
        refiner_initial.append({k:v.clone() for k,v in model.state_dict().items()})
        return model
    monkeypatch.setattr(registered,"_build_refiner",build_record)
    observed_bases=[]
    reconstruct=registered.reconstruct_from_terminal_lr
    def reconstruct_record(*a,**kw):
        assert kw["output_mode"]=="terminal_correction"
        result=reconstruct(*a,**kw)
        observed_bases.append(result.clone())
        return result
    residual=registered._physical_residual
    def residual_record(process,projector,observed,base):
        torch.testing.assert_close(base,observed_bases[-1],atol=0,rtol=0)
        return residual(process,projector,observed,base)
    monkeypatch.setattr(registered,"reconstruct_from_terminal_lr",reconstruct_record)
    monkeypatch.setattr(registered,"_physical_residual",residual_record)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    registered.train(args)
    assert len(observed_bases)==2
    best=registered._checkpoint_path(args)
    assert best.endswith("registered_new_terminal_correction.pth")
    assert registered._log_path(args).endswith("registered_new_terminal_correction.csv")
    state=torch.load(best,map_location="cpu",weights_only=False)
    assert state["epoch"]==1 and state["extra"]["diffusion_output"]=="terminal_correction"
    assert any(not torch.equal(value,refiner_initial[0][key]) for key,value in state["model"].items())
    args.refiner_checkpoint=best
    registered.test(args)
    assert len(observed_bases)==3
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    for model,old in records:
        assert all(not p.requires_grad for p in model.parameters())
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,old[key],atol=0,rtol=0)
    args.diffusion_output="state"
    wrong=registered._build_refiner(args,info,torch.device("cpu"))
    with pytest.raises(ValueError,match="diffusion_output mismatch"):
        registered._load_refiner_checkpoint(wrong,best,torch.device("cpu"))


def test_default_registered_reconstruction_is_unchanged():
    from degradations.physical import PhysicalDegradation
    from degradations.progressive import ProgressiveDegradation
    class Model(torch.nn.Module):
        def forward(self,x,t):
            return x*.9+.02
    process=ProgressiveDegradation(PhysicalDegradation(scale_ratio=4),total_steps=4)
    gt=torch.rand(1,4,32,32)
    observed,base=registered._reconstruct_base(Model(),process,gt,None)
    expected=reconstruct_from_terminal_lr(Model(),process,observed,target_size=(32,32))
    torch.testing.assert_close(base,expected,atol=0,rtol=0)

