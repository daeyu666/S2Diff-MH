"""Active capacity matching, shared training and frozen-front-end integration."""
import contextlib
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import ParameterMatchedResidualRefiner, TerminalGIGISpectralRefiner, build_terminal_refiner, RawMSIDirectPredictor
import train_innovation3_gigi as registered
import train_innovation3_gigi_cdrdi as mixed
import run_paviau_hpgc_ablation as batch_runner


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("bands,msi", [(103,4), (31,3), (144,8), (128,8), (145,8), (242,4)])
def test_exact_matching_on_all_default_dataset_architectures(bands, msi):
    control = ParameterMatchedResidualRefiner(bands, msi)
    reference = TerminalGIGISpectralRefiner(bands, msi)
    assert count(control) == count(reference) == control.match.target_params
    assert control.match.relative_error == 0.
    if bands == 103:
        assert count(control) == 544431
    assert all(p.requires_grad for p in control.parameters())
    assert not any(isinstance(m, torch.nn.MultiheadAttention) or "GlobalInterGuided" in type(m).__name__ for m in control.modules())


def test_identity_initialization_and_all_four_inputs_and_all_parameters_are_active():
    model = ParameterMatchedResidualRefiner(103, 4)
    inputs = [torch.rand(1,103,8,8), torch.rand(1,4,8,8), torch.randn(1,103,8,8)*.01, torch.rand(1,8,8)]
    output, details = model(*inputs, return_details=True)
    torch.testing.assert_close(output, inputs[0], atol=0, rtol=0)
    assert "attention" not in details and "query" not in details
    with torch.no_grad():
        model.out_head.weight.normal_(0., .01)
    inputs = [x.requires_grad_() for x in inputs]
    pred = model(*inputs)
    pred.square().mean().backward()
    for x in inputs:
        assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad), name
    update = model(*[x.detach() for x in inputs]) - inputs[0].detach()
    unit = inputs[0].detach() / torch.linalg.vector_norm(inputs[0].detach(), dim=1, keepdim=True)
    assert (update * unit).sum(1).abs().max() < 1e-5


def test_factory_preserves_existing_hpgc_weights_and_forward():
    kwargs = dict(n_bands=31, n_msi_bands=3, hidden_channels=8, heads=1, tangent_output=True)
    torch.manual_seed(10)
    original = TerminalGIGISpectralRefiner(**kwargs)
    torch.manual_seed(10)
    current = build_terminal_refiner(**kwargs)
    assert type(current) is type(original)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(current.state_dict()[key], value, atol=0, rtol=0)


def test_registered_batch_trains_only_residual_with_full_configuration_and_combines_results(tmp_path, monkeypatch):
    kwargs = ["--device","cpu","--epochs","1","--eval_interval","1","--save_interval","1",
              "--batch_size","1","--base_channels","8","--time_dim","32","--spectral_hidden","2",
              "--refine_hidden","8","--heads","1","--diffusion_steps","4",
              "--baseline_checkpoint_type","standard","--diffusion_output","terminal_correction"]
    args = registered.parse_args(kwargs)
    full = tmp_path / "full.pth"
    baseline = tmp_path / "diffusion.pth"
    args.baseline_checkpoint = str(baseline)
    reference = TerminalGIGISpectralRefiner(31,3,8,1)
    torch.save({"model":reference.state_dict(),"variant":"full","extra":vars(args)},full)
    backbone = RawMSIDirectPredictor(31,3,4,8,32,spectral_hidden=2)
    torch.save({"model":backbone.state_dict()},baseline)
    frozen_before = hashlib.sha256(baseline.read_bytes()).hexdigest()
    full_before = hashlib.sha256(full.read_bytes()).hexdigest()
    info = dict(n_bands=31,n_msi_bands=3,srf_weights=np.full((3,31),1/31))
    gt = torch.rand(31,64,64)
    loader = torch.utils.data.DataLoader([{"gt":gt,"hr_msi":gt[:3]}],batch_size=1)
    monkeypatch.setattr(registered,"build_train_val_test_loaders",lambda cfg:(loader,loader,loader,info))
    monkeypatch.setattr(registered,"build_loaders",lambda cfg:(loader,loader,info))
    monkeypatch.setattr(batch_runner,"ablation_script",lambda:ROOT/"train_innovation3_gigi.py")
    commands=[]
    def execute(command,**kw):
        commands.append(command)
        parsed=registered.parse_args(command[2:])
        assert (parsed.epochs,parsed.batch_size,parsed.lr,parsed.diffusion_output)==(1,1,2e-4,"terminal_correction")
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            (registered.train if parsed.stage=="train" else registered.test)(parsed)
        return out.getvalue()
    monkeypatch.setattr(batch_runner,"execute",execute)
    batch_runner.main(["--stage","all","--variants","residual","--full_checkpoint",str(full),
        "--baseline_checkpoint",str(baseline),"--device","cpu","--diffusion_output","terminal_correction",
        "--ablation_checkpoint_dir",str(tmp_path/"ablation"),"--log_dir",str(tmp_path/"logs"),"--result_dir",str(tmp_path/"results")])
    assert len(commands)==3
    assert [c[c.index("--variant")+1] for c in commands]==["residual","full","residual"]
    csv_path=tmp_path/"results/PaviaU_HPGC_Registered_Ablation_terminal_correction.csv"
    with csv_path.open() as stream: rows=list(csv.DictReader(stream))
    assert [r["Variant"] for r in rows]==["w/o Refiner","Residual Refiner (matched)","Full Model"]
    assert int(rows[1]["Params_refiner"])==int(rows[2]["Params_refiner"])==count(reference)
    saved=torch.load(tmp_path/"ablation/PaviaU_HPGC_Residual_Matched_terminal_correction.pth",weights_only=False)
    assert saved["variant"]=="residual" and saved["refiner_parameter_match"]["actual_params"]==count(reference)
    assert torch.count_nonzero(saved["model"]["out_head.weight"])
    assert hashlib.sha256(full.read_bytes()).hexdigest()==full_before
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==frozen_before
    # A different diffusion-output version cannot silently become the reference.
    with pytest.raises(ValueError,match="differs from the Full"):
        batch_runner.main(["--stage","test","--variants","residual","--full_checkpoint",str(full),
                           "--baseline_checkpoint",str(baseline),"--diffusion_output","state"])


def test_mixed_residual_train_and_test_freeze_both_frontends(tmp_path, monkeypatch):
    args=mixed.build_arg_parser().parse_args(["--variant","residual","--device","cpu","--from_scratch",
        "--geometry_base_channels","8","--diffusion_base_channels","8","--time_dim","32","--spectral_hidden","2",
        "--refine_hidden","8","--heads","1","--geometry_steps","1","--diffusion_steps","4","--eval_cases","1",
        "--epochs","1","--batch_size","1","--eval_interval","1","--save_interval","1",
        "--diffusion_output","terminal_correction","--checkpoint_root",str(tmp_path/"ckpt"),"--log_root",str(tmp_path/"logs")])
    info=dict(n_bands=31,n_msi_bands=3,srf_weights=np.full((3,31),1/31))
    device=torch.device("cpu")
    geometry,diffusion=mixed._build_frozen_models(args,info,device,load_weights=False)
    for name,model in (("geometry",geometry),("diffusion",diffusion)):
        path=tmp_path/(name+".pth"); torch.save({"model":model.state_dict()},path)
        setattr(args,name+"_checkpoint",str(path))
    gt=torch.rand(31,64,64)
    loader=torch.utils.data.DataLoader([{"gt":gt,"hr_msi":gt[:3]}],batch_size=1)
    monkeypatch.setattr(mixed,"build_train_val_test_loaders",lambda cfg:(loader,loader,loader,info))
    monkeypatch.setattr(mixed,"build_loaders",lambda cfg:(loader,loader,info))
    original=mixed._build_frozen_models; frozen=[]
    def build(*a,**kw):
        models=original(*a,**kw)
        frozen.extend((m,{k:v.clone() for k,v in m.state_dict().items()}) for m in models)
        return models
    monkeypatch.setattr(mixed,"_build_frozen_models",build)
    mixed.train(args)
    args.refiner_checkpoint=mixed._checkpoint_path(args)
    mixed.test(args)
    saved=torch.load(args.refiner_checkpoint,weights_only=False)
    assert saved["variant"]=="residual" and saved["extra"]["diffusion_output"]=="terminal_correction"
    for model,before in frozen:
        assert all(not p.requires_grad for p in model.parameters())
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    # Automatic mixed initialization uses the corresponding ordinary registered
    # refiner, never the full HPGC checkpoint.
    args.from_scratch=False; args.init_refiner_checkpoint=""; args.stage="train"
    mixed._resolve_dataset_checkpoints(args)
    assert args.init_refiner_checkpoint.endswith("PaviaU_innovation3_gigi_residual_terminal_correction.pth")
