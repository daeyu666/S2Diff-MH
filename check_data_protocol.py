"""Audit actual six-dataset loaders. Run from either repository, CPU only.

python check_data_protocol.py --data-root /path/to/data/raw --output audit.json
Compare two reports with --compare other.json. rasterio is required for geographic QA.
"""
import argparse, gc, hashlib, inspect, json, os
from pathlib import Path
import numpy as np
import torch
import data_loader as dl
from degradations import build_degradation
from srf_utils import sensor_protocol, load_hsi_wavelengths

NAMES = ['PaviaU','Houston13','Chikusei','CAVE','Botswana','Augsburg']
EXPECTED_BANDS = [103,144,128,31,145,242]

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def footprint(ds):
    if hasattr(ds,'samples'):
        return [list(x) for x in ds.samples]
    return [[int(t),int(l),int(ds.patch_size)] for t,l in ds.coords]

def geographic_audit(root):
    import rasterio
    files, bounds = {}, {}
    for split,folder,suffix in [('train','sr_deep_model_data','deep_train'),('validation','sr_deep_model_data','deep_valid'),('test','sub_area_1','sub_area1')]:
        for prefix,bands in [('EeteS_EnMAP_10m',242),('EeteS_EnMAP_30m',242),('EeteS_Sentinel_2_10m',4)]:
            rel=f'{folder}/{prefix}_{suffix}.tif'
            with rasterio.open(Path(root)/rel) as f:
                a=f.read()
                assert f.count==bands and np.isfinite(a).all(), rel
                files[rel]={'shape_chw':list(a.shape),'crs':str(f.crs),'bounds':list(f.bounds),'resolution':list(f.res),'range':[float(a.min()),float(a.max())], 'all_band_zero_pixels':int(np.all(a==0,axis=0).sum())}
                if prefix=='EeteS_EnMAP_10m': bounds[split]=list(f.bounds)
                assert str(f.crs)=='EPSG:32632'
                assert np.allclose(f.bounds,bounds[split])
                del a
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        l1,b1,r1,t1=bounds[a];l2,b2,r2,t2=bounds[b]
        assert max(0,min(r1,r2)-max(l1,l2))*max(0,min(t1,t2)-max(b1,b2))==0, (a,b)
    real={}
    for rel in ['entire_city/Sentinel-2.tif','sub_area_1/Sentinel_2_sub_area1.tif']:
        with rasterio.open(Path(root)/rel) as f:
            a=f.read([2,3,4,8]);assert np.isfinite(a).all()
            real[rel]={'shape_chw':[f.count,f.height,f.width],'crs':str(f.crs),'bounds':list(f.bounds),'descriptions':list(f.descriptions),'selected_1based_bands':[2,3,4,8], 'selected_range':[float(a.min()),float(a.max())]}
    return {'files':files,'split_bounds':bounds,'cross_split_intersection_m2':0,'real_sentinel2':real}

def run(args):
    from config import TrainConfig
    comparison='include_validation' in inspect.signature(dl.build_datasets).parameters
    torch.set_num_threads(2)
    report={'repository':str(Path(__file__).resolve().parent),'data_root':str(Path(args.data_root).resolve()),'settings':{'patch':64,'stride':32,'test':128,'scale':4,'srf_interp':'pchip'},'datasets':{},'resources':{}}
    for folder in ['splits','srf','wavelengths']:
        for p in sorted((Path(__file__).resolve().parent/'data'/folder).glob('*')):
            if p.is_file(): report['resources'][str(p.relative_to(Path(__file__).resolve().parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name,bands in zip(NAMES,EXPECTED_BANDS):
        print('CHECK',name,flush=True)
        cfg=TrainConfig();cfg.dataset=name;cfg.data_root=args.data_root
        assert cfg.degradation_mode=='physical'
        operator=build_degradation('physical',scale_ratio=4,mtf_nyquist=0.2,truncate=3.0)
        if comparison:
            from config import get_dataset_configs
            cfg.datasets=get_dataset_configs()
            train,val,test,info=dl.build_datasets(cfg,include_validation=True)
        else: train,val,test,info=dl.build_datasets(cfg)
        datasets=[train,val,test];assert all(len(s)>0 for s in datasets)
        waves=info.get('hsi_wavelengths')
        if waves is None: waves=load_hsi_wavelengths(sensor_protocol(name)['wavelength_path'],bands)
        w=info['srf_weights'];assert w.shape==({'PaviaU':4,'Houston13':8,'Chikusei':8,'CAVE':3,'Botswana':8,'Augsburg':4}[name],bands)
        assert np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-6,rtol=0)
        assert len(waves)==bands and np.isfinite(waves).all()
        item={'bands':bands,'wavelength_count':len(waves),'wavelength_range':[float(waves.min()),float(waves.max())],'wavelength_nonincreasing_indices':np.flatnonzero(np.diff(waves)<=0).tolist(),'srf_shape':list(w.shape),'srf_row_sums':w.sum(1).tolist(),'srf_sha256':digest(w),'counts':[len(s) for s in datasets], 'split_coordinate_sha256':[], 'sample_hashes':[], 'sample_shapes':[], 'terminal_hsi_hashes':[]}
        if name=='CAVE':
            mapping=dl._find_cave_scene_dirs(str(Path(args.data_root)/'CAVE'))
            shapes={}; encodings={}
            for scene,path in sorted(mapping.items()):
                cube=dl._load_cave_scene(path)
                assert cube.shape==(512,512,31) and np.isfinite(cube).all()
                shapes[scene]=list(cube.shape)
                first=next(Path(path).glob('*_ms_01.png'))
                header=first.read_bytes()[:29]
                encodings[scene]={'bit_depth':header[24],'png_color_type':header[25]}
            sets=[{x[0] for x in s.samples} for s in datasets]
            assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
            item.update(raw_shape=[32,512,512,31],prepared_shape=[512,512,31],scene_shapes=shapes,scene_encodings=encodings,scene_counts=[len(s) for s in sets],cross_split_overlap=0)
        else:
            images=[getattr(s,'image',None) if hasattr(s,'image') else s.img for s in datasets]
            assert all(a.shape[2]==bands and np.isfinite(a).all() for a in images)
            item['prepared_shapes']=[list(a.shape) for a in images]
            if name=='Augsburg':
                item['geographic']=geographic_audit(dl._find_augsburg_root(args.data_root))
                item['raw_shape']=item['prepared_shapes']
                item['evaluation_covered_pixels']=[len(s)*s.patch_size**2 for s in datasets[1:]]
            else:
                path=Path(args.data_root)/(name+'.mat')
                import scipy.io,h5py
                if h5py.is_hdf5(path):
                    with h5py.File(path) as f:
                        obj=next(v for v in f.values() if isinstance(v,h5py.Dataset) and len(v.shape)==3)
                        item['raw_shape']=list(reversed(obj.shape)) if 'MATLAB_class' in obj.attrs else list(obj.shape)
                        if name=='Chikusei':
                            # Verify MATLAB axes and the first cropped pixel independently.
                            assert item['raw_shape']==[2517,2335,128]
                            item['crop_origin']=[106,143]
                else: item['raw_shape']=next(list(shape) for key,shape,kind in scipy.io.whosmat(path) if len(shape)==3)
                masks=[]
                for s in datasets:
                    mask=np.zeros(images[0].shape[:2],dtype=bool)
                    for top,left in s.coords: mask[top:top+s.patch_size,left:left+s.patch_size]=True
                    masks.append(mask)
                for i,j in [(0,1),(0,2),(1,2)]: assert not np.any(masks[i]&masks[j]), (name,i,j)
                item['validation_rect']=list(info['validation_rect']);item['test_rect']=list(info['test_rect'])
            item['cross_split_overlap']=0
        for s in datasets:
            coords=footprint(s);item['split_coordinate_sha256'].append(hashlib.sha256(json.dumps(coords).encode()).hexdigest())
            s.augment=False
            indices=sorted(set([0,len(s)-1])) if s.split=='train' else list(range(len(s)))
            hashes=[]; shapes=[]; terminal_hashes=[]
            for index in indices:
                x=s[index]; assert torch.isfinite(x['gt']).all() and torch.isfinite(x['hr_msi']).all()
                assert x['gt'].shape[0]==bands and x['hr_msi'].shape[0]==w.shape[0]
                projected=torch.einsum('mc,chw->mhw',torch.from_numpy(w),x['gt'])
                assert torch.allclose(projected,x['hr_msi'],atol=1e-6,rtol=1e-5)
                hashes.append({k:digest(x[k].numpy()) for k in ['gt','hr_msi']})
                shapes.append({k:list(v.shape) for k,v in x.items() if k in ['gt','hr_msi','lr_hsi']})
                terminal=operator.degrade(x['gt'].unsqueeze(0)).squeeze(0)
                terminal_hashes.append(digest(terminal.numpy()))
                if comparison:
                    assert torch.allclose(terminal,x['lr_hsi'],atol=1e-6,rtol=0)
                    y=s.degradation_operator.degrade(x['gt'].unsqueeze(0)).squeeze(0)
                    assert torch.allclose(y,x['lr_hsi'],atol=1e-6,rtol=0)
            item['sample_hashes'].append(hashes);item['sample_shapes'].append(shapes);item['terminal_hsi_hashes'].append(terminal_hashes)
        report['datasets'][name]=item
        Path(args.output).write_text(json.dumps(report,indent=2))
        print('PASS',name,item['raw_shape'],item['counts'],flush=True)
        del train,val,test,datasets,info;gc.collect()
    if args.compare:
        other=json.loads(Path(args.compare).read_text())
        for path,sha in other['resources'].items():
            assert report['resources'].get(path)==sha, path
        for name in NAMES:
            for k in ['raw_shape','counts','srf_sha256','wavelength_count','split_coordinate_sha256','sample_hashes','terminal_hsi_hashes']:
                assert report['datasets'][name][k]==other['datasets'][name][k], (name,k)
        report['cross_repository_match']=True
    report['status']='PASS';Path(args.output).write_text(json.dumps(report,indent=2));print('ALL PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='./data/raw');p.add_argument('--output',default='data_audit.json');p.add_argument('--compare');args=p.parse_args()
    run(args)
