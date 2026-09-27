import numpy as np
import pytest
import data_loader as dl
from srf_utils import estimate_band_widths, sensor_protocol, build_srf_weights

def test_matlab_v73_restores_both_spatial_axes(tmp_path, monkeypatch):
    import h5py
    expected=np.arange(300*280*7,dtype=np.float32).reshape(300,280,7)
    path=tmp_path/'scene.mat'
    with h5py.File(path,'w') as f:
        d=f.create_dataset('cube',data=expected.transpose(2,1,0));d.attrs['MATLAB_class']=np.bytes_('single')
    monkeypatch.setattr(dl,'hdf5storage',None)
    assert np.array_equal(dl.read_hsi_mat(str(path),['cube']),expected)

def test_hdf5_nonmatlab_chw_is_not_reversed(tmp_path, monkeypatch):
    import h5py
    expected=np.arange(300*280*7,dtype=np.float32).reshape(300,280,7)
    path=tmp_path/'scene.mat'
    with h5py.File(path,'w') as f: f.create_dataset('cube',data=expected.transpose(2,0,1))
    monkeypatch.setattr(dl,'hdf5storage',None)
    assert np.array_equal(dl.read_hsi_mat(str(path),['cube']),expected)

def test_overlap_grid_is_order_invariant_and_shares_duplicates():
    wave=np.array([400,410,420,405,410,415],dtype=np.float32)
    widths=estimate_band_widths(wave)
    assert np.all(widths>0)
    assert widths[1]==widths[4]
    assert np.isclose(widths.sum(),25)
    order=np.argsort(wave)
    assert np.allclose(widths[order],estimate_band_widths(wave[order]))

def test_srf_resolves_outside_repo_and_is_finite(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    from srf_utils import load_hsi_wavelengths
    p=sensor_protocol('Botswana');wave=load_hsi_wavelengths(p['wavelength_path'],145)
    w,names=build_srf_weights(p['srf_path'],wave,p['bands'])
    assert w.shape==(8,145) and np.all(w>=0)
    assert np.allclose(w.sum(1),1,atol=1e-6,rtol=0)

def test_chikusei_crop_uses_raw_scene_origin():
    # Broadcast a coordinate-coded plane; no large 128-band fixture needed.
    raw=np.broadcast_to(np.arange(2335,dtype=np.float32)[None,:,None],(2517,2335,1))
    if hasattr(dl,'_prepare_chikusei'): out=dl._prepare_chikusei(raw)
    else: out=dl._center_crop(raw,2304,2048)
    assert out.shape==(2304,2048,1) and out[0,0,0]==143


def test_cave_8bit_rgba_grayscale_and_16bit_have_correct_scale(tmp_path):
    from PIL import Image
    gray=np.array([[0,128],[255,64]],dtype=np.uint8)
    rgba=np.concatenate([np.repeat(gray[...,None],3,axis=2),np.full((2,2,1),255,dtype=np.uint8)],axis=2)
    p=tmp_path/'8.png';Image.fromarray(rgba).save(p)
    assert np.allclose(dl._read_cave_band(str(p)),gray/255)
    p=tmp_path/'16.png';Image.fromarray((gray.astype(np.uint16)*257)).save(p)
    assert np.allclose(dl._read_cave_band(str(p)),gray/255)
    rgba[0,0,1]=1;p=tmp_path/'color.png';Image.fromarray(rgba).save(p)
    with pytest.raises(ValueError,match='not replicated grayscale'): dl._read_cave_band(str(p))
