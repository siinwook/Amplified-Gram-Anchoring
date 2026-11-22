"""Final frozen-backbone probe evaluation extracted from the cleaned full-split notebook.

The original evaluation protocol is preserved. Run only after setting MODEL_PATHS,
DINOv2 dependencies and benchmark data access in the configuration below.
"""
from pathlib import Path
import os, sys, gc, copy, math, json, random, shutil, subprocess, time, hashlib
from dataclasses import dataclass, asdict
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torchvision.datasets import VOCSegmentation, OxfordIIITPet, CIFAR100, Flowers102
from torchvision.transforms import functional as TF, InterpolationMode
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from tqdm.auto import tqdm

ROOT = Path('/kaggle/working' if Path('/kaggle/working').exists() else Path.home() / 'aga')
DATA_ROOT = ROOT / 'datasets_full'
OUT = ROOT / 'AGA_Full_Eval'; OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# DATA ACCESS ONLY: local cache / attached Kaggle Input first, HF mirrors otherwise.
# No changes to original model, hyperparameters, transforms or probe selection.
VOC_LOCAL_ROOT = None   # e.g. '/kaggle/input/my-voc2012' (root containing VOCdevkit/)
PET_LOCAL_ROOT = None   # e.g. '/kaggle/input/my-oxford-pet' (root containing oxford-iiit-pet/)
VOC_SOURCE = 'auto'      # auto | local | hf
PET_SOURCE = 'auto'      # auto | local | hf
FLOWERS_LOCAL_ROOT = None  # e.g. '/kaggle/input/my-flowers102' containing flowers-102/jpg, imagelabels.mat, setid.mat
FLOWERS_SOURCE = 'auto'  # auto | local | hf; never use Oxford download=True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE', DEVICE, '| torch',torch.__version__, '| torchvision',torchvision.__version__)

# >>> EDIT these paths (one or many). You may use pretrained via PRETRAINED_REFERENCE=True.
MODEL_PATHS = {
    'pretrained': '/kaggle/input/models/YOUR_KAGGLE_USERNAME/dinov2-vits14-pretrain/pytorch/default/1/dinov2_vits14_pretrain.pth',
    'none': '/kaggle/input/models/YOUR_KAGGLE_USERNAME/none-1500iter/pytorch/default/1/student_step_01500.pt',
    'gram': '/kaggle/input/models/YOUR_KAGGLE_USERNAME/gram-1500iter/pytorch/default/1/student_step_01500.pt',
    'aga_1500iter_H10_b04_a8': '/kaggle/input/models/YOUR_KAGGLE_USERNAME/aga-1500iter-h10-b04-a8/pytorch/default/1/aga_student_step_01500_H10_b04_a8.pt',
    'aga_1500iter_H0_b04_a8': '/kaggle/input/models/YOUR_KAGGLE_USERNAME/aga-1500iter-h0-b04-a8/pytorch/default/1/aga_student_step_01500_H0_b04_a8.pt',
}
PRETRAINED_REFERENCE = False  # True adds official pretrained model as 'pretrained'
TASKS = ['voc', 'ade20k', 'pet', 'nyu', 'cifar100', 'flowers102']

@dataclass
class CFG:
    seed: int = 3407
    num_workers: int = 2
    seg_epochs: int = 20
    depth_epochs: int = 20
    cls_epochs: int = 20
    lr: float = 1e-3
    eta_min: float = 5e-5
    weight_decay: float = 1e-4
    seg_hw: tuple = (448,448)
    nyu_hw: tuple = (294,392)
    cls_hw: tuple = (224,224)
    seg_bs: int = 64
    depth_bs: int = 64
    cls_bs: int = 64
    valid_fraction: float = 0.1
    save_heads: bool = True
    refit_full_train: bool = True

cfg = CFG()
(OUT/'config.json').write_text(json.dumps({**asdict(cfg), 'tasks':TASKS, 'models':MODEL_PATHS}, indent=2, default=str))
for name,path in MODEL_PATHS.items():
    assert Path(path).is_file(), f'Missing {name}: {path} — set MODEL_PATHS first'
if not MODEL_PATHS and not PRETRAINED_REFERENCE: raise ValueError('Specify at least one model')

DINOV2_REPO = ROOT / 'dinov2'
if not (DINOV2_REPO/'hubconf.py').exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/facebookresearch/dinov2.git',str(DINOV2_REPO)],check=True)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def unpack_state(obj):
    if isinstance(obj,dict) and any(torch.is_tensor(x) for x in obj.values()):
        # Direct tensor dictionary OR mixed checkpoint tensor dictionary.
        tensor_part = {k:v for k,v in obj.items() if isinstance(k,str) and torch.is_tensor(v)}
        if len(tensor_part)>=10:return tensor_part
    if isinstance(obj,dict):
        for k in ['student','student_backbone','backbone','state_dict','model','teacher']:
            if k in obj:
                result=unpack_state(obj[k])
                if result is not None:return result
    return None

def remove_prefix(k):
    for _ in range(10):
        matched=False
        for pre in ['module.','_orig_mod.','student.backbone.','student.','model.','backbone.']:
            if k.startswith(pre):k=k[len(pre):];matched=True;break
        if not matched:break
    return k

def build_backbone(path=None):
    m = torch.hub.load(str(DINOV2_REPO),'dinov2_vits14',source='local',pretrained=(path is None))
    if path is not None:
        obj=torch.load(str(path),map_location='cpu',weights_only=False)
        state=unpack_state(obj)
        if state is None:raise RuntimeError(f'Backbone state_dict not found in {path}')
        target=m.state_dict()
        clean={remove_prefix(k):v for k,v in state.items() if remove_prefix(k) in target and v.shape == target[remove_prefix(k)].shape}
        missing,unexpected=m.load_state_dict(clean,strict=False)
        coverage=len(clean)/len(target)
        print(Path(path).name,'backbone tensor coverage',f'{coverage:.2%}', 'missing',missing[:6])
        if coverage < 0.98: raise RuntimeError(f'Checkpoint loaded only {coverage:.1%} backbone tensors')
    m.eval().to(DEVICE)
    for p in m.parameters():p.requires_grad_(False)
    return m

@torch.no_grad()
def feature_map(backbone, x, kind):
    if kind=='seg':
        out=backbone.forward_features(x)['x_norm_patchtokens']
        b,n,c=out.shape; hh,ww=x.shape[-2]//14,x.shape[-1]//14
        assert n==hh*ww
        return out.transpose(1,2).reshape(b,c,hh,ww)
    if kind=='depth':
        return torch.cat(list(backbone.get_intermediate_layers(x,n=4,reshape=True,norm=True)),dim=1)
    if kind=='cls':
        layers=backbone.get_intermediate_layers(x,n=4,return_class_token=True,norm=True)
        return torch.cat([a[1] for a in layers]+[layers[-1][0].mean(dim=1)],dim=1)
    raise ValueError(kind)

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

print('Backbone preparation ready')

# Official benchmark download is supported for ADE20K. Other datasets handled by torchvision.
ADE_ROOT = DATA_ROOT/'ADEChallengeData2016'
ADE_ZIP = DATA_ROOT/'ADEChallengeData2016.zip'
ADE_URL = 'https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
DOWNLOAD_ADE = True # official archive; >1 GB, requires connectivity (False if already downloaded)
if DOWNLOAD_ADE and 'ade20k' in TASKS and not (ADE_ROOT/'images/training').exists():
    import urllib.request, zipfile
    if not ADE_ZIP.exists(): urllib.request.urlretrieve(ADE_URL,ADE_ZIP)
    with zipfile.ZipFile(ADE_ZIP) as z:z.extractall(DATA_ROOT)


IM_MEAN=(0.485,0.456,0.406); IM_STD=(0.229,0.224,0.225)
def image_norm(img):return TF.normalize(TF.to_tensor(img),IM_MEAN,IM_STD)

class SegTF:
    def __init__(self, train, hw):self.train,self.hw=train,hw
    def __call__(self,img,mask):
        img=TF.resize(img,self.hw,interpolation=InterpolationMode.BICUBIC,antialias=True)
        mask=TF.resize(mask,self.hw,interpolation=InterpolationMode.NEAREST)
        if self.train and random.random()<.5:img,mask=TF.hflip(img),TF.hflip(mask)
        return image_norm(img), torch.from_numpy(np.asarray(mask,dtype=np.int64).copy()).long()

class ADEFull(Dataset):
    def __init__(self,split,train):
        part={'train':'training','val':'validation'}[split]
        images=ADE_ROOT/'images'/part; labels=ADE_ROOT/'annotations'/part
        assert images.exists() and labels.exists(), f'Official ADE20K missing: {images}. Set DOWNLOAD_ADE=True'
        self.samples=[]
        for p in sorted(images.glob('*.jpg')):
            q=labels/(p.stem+'.png')
            if not q.is_file():raise FileNotFoundError(q)
            self.samples.append((p,q))
        expected=20210 if split=='train' else 2000
        if len(self.samples)!=expected:raise ValueError(f'ADE {split}: {len(self.samples)} != {expected} (full benchmark)')
        self.tf=SegTF(train,cfg.seg_hw)
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        a,b=self.samples[i]
        with Image.open(a) as im:im=im.convert('RGB')
        with Image.open(b) as m:m=m.copy()
        im,m=self.tf(im,m)
        # Original mask: 0=ignore; 1..150 => 0..149
        m=torch.where(m==0,255,m-1)
        return im,m

class PetTF:
    def __init__(self,train):self.tf=SegTF(train,cfg.seg_hw)
    def __call__(self,im,m):
        x,y=self.tf(im,m)
        # Oxford trimap: 1=pet, 2=background, 3=boundary. All three are valid classes.
        if not torch.all((y>=1)&(y<=3)):raise ValueError('Unexpected Oxford Pet trimap values')
        y=torch.where(y==1,1,torch.where(y==2,0,2)).long()
        return x,y

class NYUFull(Dataset):
    def __init__(self,split,train):
        from datasets import load_dataset
        # Keep exact previous AGA source and labels.
        global _NYU_HF_CACHE
        if '_NYU_HF_CACHE' not in globals(): _NYU_HF_CACHE=load_dataset('tanganke/nyuv2')
        self.ds=_NYU_HF_CACHE[split].with_format('numpy');self.train=train
        exp=795 if split=='train' else 654
        if len(self.ds)!=exp:raise ValueError(f'NYUv2 {split}: got {len(self.ds)} expected {exp}')
    def __len__(self):return len(self.ds)
    def __getitem__(self,i):
        a=self.ds[i];img=np.asarray(a['image']);dep=np.asarray(a['depth'])
        if img.ndim==3 and img.shape[0]==3:img=np.transpose(img,(1,2,0))
        if dep.ndim==3:dep=dep[0] if dep.shape[0]==1 else dep[...,0]
        if img.max()<=1.5:img=np.clip(img*255,0,255).astype(np.uint8)
        else:img=img.astype(np.uint8)
        return preprocess_depth_pair(Image.fromarray(img,'RGB'),dep.astype('float32'),self.train,cfg.nyu_hw)

def preprocess_depth_pair(img, depth, train, hw):
    # Align original image/depth arrays and use validity-weighted resize to avoid polluting depth near sparse zeros.
    depth=np.asarray(depth,dtype=np.float32)
    if depth.ndim!=2:raise ValueError('Depth must be H×W')
    if img.size!=(depth.shape[1],depth.shape[0]):raise ValueError(f'Image and depth misaligned: {img.size} vs {depth.shape}')
    if train and random.random()<.5:
        img=TF.hflip(img); depth=np.ascontiguousarray(depth[:,::-1])
    img=TF.resize(img,hw,InterpolationMode.BICUBIC,antialias=True)
    x=image_norm(img)
    dep=torch.from_numpy(depth.copy()).float()[None,None]
    mask=(torch.isfinite(dep)&(dep>1e-3)).float()
    dep=torch.nan_to_num(dep,nan=0.0,posinf=0.0,neginf=0.0)*mask
    weighted=F.interpolate(dep,size=hw,mode='bilinear',align_corners=False)
    weight=F.interpolate(mask,size=hw,mode='bilinear',align_corners=False)
    dep=torch.where(weight>0.5,weighted/weight.clamp_min(1e-6),torch.zeros_like(weighted))
    return x,dep[0]

cls_train_transform=T.Compose([T.Resize(cfg.cls_hw,InterpolationMode.BICUBIC),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(IM_MEAN,IM_STD)])
cls_eval_transform=T.Compose([T.Resize(cfg.cls_hw,InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(IM_MEAN,IM_STD)])

# -----------------------------------------------------------------------------
# Stable, Oxford-server-independent dataset adapters for VOC2012 & Oxford Pet.
# 'auto': verified local torchvision-style data -> entire HF public split.
# The HF options are not arbitrary subsets. Verify full official row counts.
# This change affects *only* data acquisition, not transforms or metrics.
# -----------------------------------------------------------------------------

def _valid_voc_root(root):
    base=Path(root)/'VOCdevkit'/'VOC2012'
    try:
        a=(base/'ImageSets/Segmentation/train.txt').read_text().splitlines()
        b=(base/'ImageSets/Segmentation/val.txt').read_text().splitlines()
        if len(a)!=1464 or len(b)!=1449:return False
        images=base/'JPEGImages'; masks=base/'SegmentationClass'
        return (images.is_dir() and masks.is_dir() and
                all((images/(k.strip()+'.jpg')).is_file() and
                    (masks/(k.strip()+'.png')).is_file() for k in a+b))
    except (FileNotFoundError, PermissionError, UnicodeError):return False

def _valid_pet_root(root):
    base=Path(root)/'oxford-iiit-pet'
    im=base/'images'; ann=base/'annotations'; tri=ann/'trimaps'
    if not (im.is_dir() and tri.is_dir()):return False
    for split,n in [('trainval',3680),('test',3669)]:
        p=ann/(split+'.txt')
        if not p.is_file():return False
        lines=p.read_text().splitlines()
        try: ids=[a.split()[0] for a in lines if a.strip() and not a.startswith('#')]
        except Exception:return False
        if len(ids)!=n:return False
        if not all((im/(k+'.jpg')).is_file() and (tri/(k+'.png')).is_file() for k in ids):return False
    return True

def _candidate_roots(task,manual):
    # Root is the parent of VOCdevkit/ or oxford-iiit-pet/.
    result=[]
    if manual:result.append(Path(manual))
    result.append(DATA_ROOT/('voc' if task=='voc' else 'pet'))
    base=Path('/kaggle/input')
    if base.is_dir():
        # Probe 1-2 directory levels without crawling entire datasets.
        patterns=('*/VOCdevkit/VOC2012','*/*/VOCdevkit/VOC2012') if task=='voc' else ('*/oxford-iiit-pet','*/*/oxford-iiit-pet')
        for pat in patterns:
            for match in base.glob(pat):
                result.append(match.parent.parent if task=='voc' else match.parent)
    seen=set()
    for p in result:
        key=str(p.resolve())
        if key not in seen:
            seen.add(key); yield p

class _HFSegmentation(Dataset):
    def __init__(self, ds, task, train):
        self.ds=ds;self.task=task
        self.transform=SegTF(train,cfg.seg_hw) if task=='voc' else PetTF(train)
    def __len__(self):return len(self.ds)
    def __getitem__(self,i):
        row=self.ds[int(i)]
        if self.task=='voc':
            img, mask=row['image'],row['mask']
        else:
            img,mask=row['image'],row['segmentation_mask']
        if not isinstance(img,Image.Image):img=Image.fromarray(np.asarray(img).astype(np.uint8))
        if not isinstance(mask,Image.Image):
            arr=np.asarray(mask)
            if arr.ndim==3 and arr.shape[2]==1:arr=arr[:,:,0]
            if arr.ndim!=2:raise ValueError(f'{self.task}: expected class-index mask, got {arr.shape}')
            mask=Image.fromarray(arr.astype(np.uint8))
        # Never call mask.convert('RGB'): class indices would be destroyed.
        return self.transform(img.convert('RGB'),mask)

_HF_SOURCE_CACHE={}
def _load_hf_public_split(dataset_name, config=None):
    from datasets import load_dataset
    key=(dataset_name,config)
    if key not in _HF_SOURCE_CACHE:
        print('[DATA] Loading standard dataset from Hugging Face:',dataset_name,
              '\n       This may take time on first run; never use subset/shuffle to reduce the benchmark.')
        opts=dict(cache_dir=str(DATA_ROOT/'hf_datasets_cache'))
        _HF_SOURCE_CACHE[key]=load_dataset(dataset_name,config,**opts) if config else load_dataset(dataset_name,**opts)
    return _HF_SOURCE_CACHE[key]

def _voc_full():
    source=VOC_SOURCE.lower()
    if source not in ('auto','local','hf'):raise ValueError('VOC_SOURCE must be auto/local/hf')
    if source in ('auto','local'):
        for root in _candidate_roots('voc',VOC_LOCAL_ROOT):
            if _valid_voc_root(root):
                print('[DATA] VOC2012 local official files:',root,'(download=False)')
                tr=VOCSegmentation(str(root),year='2012',image_set='train',download=False,transforms=SegTF(True,cfg.seg_hw))
                va=VOCSegmentation(str(root),year='2012',image_set='val',download=False,transforms=SegTF(False,cfg.seg_hw))
                assert len(tr)==1464 and len(va)==1449
                return tr,va,'official local VOCdevkit'
        if source=='local':raise FileNotFoundError('VOC2012 1464/1449 official segmentation files not found. Set VOC_LOCAL_ROOT or VOC_SOURCE="hf".')
    try:
        # HF mirror preserves official VOC 2012 segmentation masks (0..20, 255=ignore), no SBD augmentation.
        data=_load_hf_public_split('shijli/voc2012','segmentation')
        val=data['val'] if 'val' in data else data['validation']
        assert len(data['train'])==1464 and len(val)==1449, ('Unexpected VOC HF split sizes',len(data['train']),len(val))
        sample=np.asarray(data['train'][0]['mask'])
        if sample.ndim!=2 or np.any((sample>20)&(sample!=255)):
            raise ValueError('VOC HF masks not raw 0..20/255 labels; refusing to change mIoU protocol')
        print('[DATA] VOC2012 official 1464/1449 from HF segmentation mirror; no Oxford site requests')
        return _HFSegmentation(data['train'],'voc',True),_HFSegmentation(val,'voc',False),'HF shijli/voc2012 segmentation (official splits)'
    except Exception as e:
        raise RuntimeError('VOC2012 could not be loaded from local files or HF mirror. '
                           'On Kaggle, enable Internet or attach VOC2012 as Kaggle Input and set VOC_LOCAL_ROOT. '
                           'Original Oxford download is intentionally bypassed to avoid hanging.') from e

def _pet_full():
    source=PET_SOURCE.lower()
    if source not in ('auto','local','hf'):raise ValueError('PET_SOURCE must be auto/local/hf')
    if source in ('auto','local'):
        for root in _candidate_roots('pet',PET_LOCAL_ROOT):
            if _valid_pet_root(root):
                print('[DATA] Oxford Pet local official files:',root,'(download=False)')
                trall=OxfordIIITPet(str(root),split='trainval',target_types='segmentation',download=False,transforms=PetTF(True))
                valall=OxfordIIITPet(str(root),split='trainval',target_types='segmentation',download=False,transforms=PetTF(False))
                test=OxfordIIITPet(str(root),split='test',target_types='segmentation',download=False,transforms=PetTF(False))
                assert len(trall)==3680 and len(test)==3669
                return trall,valall,test,'official local Oxford-IIIT Pet trainval/test'
        if source=='local':raise FileNotFoundError('Oxford Pet official trainval/test images/trimaps missing. Set PET_LOCAL_ROOT or PET_SOURCE="hf".')
    try:
        # timm Pet HF mirror DOES NOT provide trimaps. Use a mirror that DOES.
        data=_load_hf_public_split('dpdl-benchmark/oxford_iiit_pet')
        assert len(data['train'])==3680 and len(data['test'])==3669, ('Unexpected Oxford Pet split sizes',len(data['train']),len(data['test']))
        # Strictly check raw segmentation label encoding before using existing PetTF mapping.
        values=set()
        for j in [0, 1, 7, 71, 137]:
            m=np.asarray(data['train'][j]['segmentation_mask'])
            values.update(int(x) for x in np.unique(m))
        if not values.issubset({1,2,3}) or not values:
            raise ValueError(f'Pet mirror label coding {sorted(values)} differs from official 1=pet,2=background,3=boundary; '
                             'provide official local trimaps instead. No automatic remapping.')
        trall=_HFSegmentation(data['train'],'pet',True)
        valall=_HFSegmentation(data['train'],'pet',False)
        test=_HFSegmentation(data['test'],'pet',False)
        print('[DATA] Oxford Pet full trainval=3680 test=3669 from HF mirror WITH trimaps')
        return trall,valall,test,'HF dpdl-benchmark/oxford_iiit_pet (official split lengths, mask labels verified)'
    except Exception as e:
        raise RuntimeError('Oxford-IIIT Pet could not be loaded from local files or HF segmentation mirror. '
                           'On Kaggle, enable Internet or attach the full Oxford Pet dataset with images AND trimaps '
                           'as Kaggle Input; set PET_LOCAL_ROOT. The Oxford website is intentionally bypassed.') from e

# Oxford Flowers-102: local official torchvision dataset or split-faithful HF mirror.
# Explicitly never call Flowers102(..., download=True): Oxford's remote host can time out.
# The benchmark's canonical split sizes are train=1020, val=1020, test=6149.

def _flowers_local_roots():
    paths=[]
    if FLOWERS_LOCAL_ROOT: paths.append(Path(FLOWERS_LOCAL_ROOT))
    paths.append(DATA_ROOT/'flowers102')
    inp=Path('/kaggle/input')
    if inp.is_dir():
        # Kaggle Input: look for official torchvision-compatible flowers-102 directory.
        for pattern in ('*/flowers-102', '*/*/flowers-102', '*/*/*/flowers-102'):
            paths.extend(m.parent for m in inp.glob(pattern))
    seen=set()
    for p in paths:
        key=str(p.resolve())
        if key not in seen:
            seen.add(key)
            yield p

class _HFFlowers102(Dataset):
    def __init__(self, ds, train):
        self.ds=ds
        self.tf=cls_train_transform if train else cls_eval_transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, index):
        row=self.ds[int(index)]
        image=row['image']
        if not isinstance(image, Image.Image):
            image=Image.fromarray(np.asarray(image).astype(np.uint8))
        label=int(row['label'])
        if not (0<=label<102):
            raise ValueError(f'Flowers102 label outside 0..101: {label}')
        return self.tf(image.convert('RGB')), label

def _flowers_full():
    source=FLOWERS_SOURCE.lower()
    if source not in ('auto','local','hf'):
        raise ValueError('FLOWERS_SOURCE must be auto/local/hf')
    if source in ('auto','local'):
        for root in _flowers_local_roots():
            try:
                # download=False is essential: never contact the Oxford server.
                train=Flowers102(str(root), split='train', download=False, transform=cls_train_transform)
                val=Flowers102(str(root), split='val', download=False, transform=cls_eval_transform)
                test=Flowers102(str(root), split='test', download=False, transform=cls_eval_transform)
                if (len(train),len(val),len(test))!=(1020,1020,6149):
                    raise ValueError('Nonstandard Flowers102 split lengths')
                # Check actual files for all official samples, not just their parent folder.
                if not all(f.is_file() for f in (train._image_files+val._image_files+test._image_files)):
                    raise FileNotFoundError('Incomplete Flowers102 jpg directory')
                val_train=Flowers102(str(root), split='val', download=False, transform=cls_train_transform)
                full=ConcatDataset([train,val_train])
                print('[DATA] Oxford Flowers102: local verified 1020/1020/6149; download=False:', root)
                return train,val,test,full,'local torchvision Flowers102 (official splits)'
            except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
                print('[DATA] Skipping missing/incomplete local Flowers102:',root, str(exc)[:110])
        if source=='local':
            raise FileNotFoundError('Full Flowers102 not found locally. Attach Kaggle Input with '
                                    'flowers-102/jpg, flowers-102/imagelabels.mat, flowers-102/setid.mat '
                                    'and set FLOWERS_LOCAL_ROOT to its parent; or use FLOWERS_SOURCE="hf".')
    try:
        # This verified mirror has the official Oxford split (unlike custom 80/10/10 mirrors).
        data=_load_hf_public_split('dpdl-benchmark/oxford_flowers102')
        assert all(x in data for x in ('train','validation','test')), 'Missing standard split keys'
        lengths=(len(data['train']),len(data['validation']),len(data['test']))
        if lengths!=(1020,1020,6149):
            raise ValueError(f'Flowers102 mirror uses nonstandard splits: {lengths}')
        for key in ('train','validation','test'):
            ds=data[key]
            if 'image' not in ds.column_names or 'label' not in ds.column_names:
                raise ValueError(f'Flowers102 {key}: missing image/label columns')
            feature=ds.features['label']
            if hasattr(feature,'num_classes') and feature.num_classes!=102:
                raise ValueError(f'Flowers102 {key}: unexpected class count {feature.num_classes}')
            labels=ds['label']
            if min(labels)<0 or max(labels)>=102:
                raise ValueError(f'Flowers102 {key}: invalid label range')
        train=_HFFlowers102(data['train'],True)
        val=_HFFlowers102(data['validation'],False)
        test=_HFFlowers102(data['test'],False)
        val_train=_HFFlowers102(data['validation'],True)
        full=ConcatDataset([train,val_train])
        print('[DATA] Oxford Flowers102: HF official split 1020/1020/6149; full refit=2040')
        return train,val,test,full,'HF dpdl-benchmark/oxford_flowers102 (official split, verified counts and labels)'
    except Exception as exc:
        raise RuntimeError('Oxford Flowers102 cannot be loaded from the Hugging Face mirror. '
                           'If Kaggle network/CDN also fails, attach official Flowers102 as Kaggle Input '
                           '(flowers-102/jpg + imagelabels.mat + setid.mat), then set '
                           'FLOWERS_LOCAL_ROOT and FLOWERS_SOURCE="local". '
                           'No Oxford remote downloads are attempted by this notebook.') from exc

def get_dataset(task):
    # Returns training, validation, held-out test (or None), class count, kind, metadata
    if task=='voc':
        tr,va,origin=_voc_full()
        return tr,va,None,21,'seg',{'split':'VOC train/val','selection_split':'val','data_source':origin}
    if task=='ade20k':return ADEFull('train',True),ADEFull('val',False),None,150,'seg',{'split':'ADE SceneParsing train/val','selection_split':'val'}
    if task=='pet':
        # Retain exact seed-based 90/10 holdout and trainval full-data refit.
        trall,valall,test,origin=_pet_full()
        ix=np.random.default_rng(cfg.seed).permutation(len(trall)); vn=max(1,round(len(ix)*cfg.valid_fraction)); tidx=ix[vn:].tolist();vidx=ix[:vn].tolist()
        return Subset(trall,tidx),Subset(valall,vidx),test,3,'seg',{'split':'pet trainval internal holdout + official test','selection_split':'internal_val','full_train':trall,'data_source':origin}
    if task=='nyu':return NYUFull('train',True),NYUFull('val',False),None,1,'depth',{'split':'NYUv2 795/654','selection_split':'val'}
    if task=='cifar100':
        alltrain=CIFAR100(str(DATA_ROOT/'cifar100'),train=True,download=True,transform=cls_train_transform)
        valall=CIFAR100(str(DATA_ROOT/'cifar100'),train=True,download=True,transform=cls_eval_transform)
        test=CIFAR100(str(DATA_ROOT/'cifar100'),train=False,download=True,transform=cls_eval_transform)
        # Stratified 10% of *official* train for epoch selection only.
        labels=np.asarray(alltrain.targets);trix=[];vix=[];rng=np.random.default_rng(cfg.seed)
        for y in np.unique(labels):
            ix=np.flatnonzero(labels==y);rng.shuffle(ix);n=max(1,round(len(ix)*cfg.valid_fraction));vix.extend(ix[:n]);trix.extend(ix[n:])
        return Subset(alltrain,trix),Subset(valall,vix),test,100,'cls',{'split':'CIFAR100 full train/test, internal stratified validation','selection_split':'internal_val','full_train':alltrain}
    if task=='flowers102':
        tr,va,test,full,origin=_flowers_full()
        return tr,va,test,102,'cls',{'split':'Oxford Flowers102 official train/val/test',
                                     'selection_split':'val','full_train':full,'data_source':origin}
    raise KeyError(task)

print('Dataset adapters ready. Dataset access occurs one task at a time.')

def loader(dataset,train,kind,task,seed_offset=0):
    bs = (cfg.seg_bs if kind=='seg' else cfg.depth_bs if kind=='depth' else cfg.cls_bs)
    g=torch.Generator().manual_seed(cfg.seed+seed_offset)
    return DataLoader(dataset,batch_size=bs,shuffle=train,num_workers=cfg.num_workers,pin_memory=torch.cuda.is_available(),drop_last=False,persistent_workers=cfg.num_workers>0,generator=g)

class Probe(nn.Module):
    def __init__(self,kind,nclasses):
        super().__init__();self.kind=kind
        if kind=='seg':self.layer=nn.Conv2d(384,nclasses,1)
        elif kind=='depth':
            self.layer=nn.Conv2d(384*4,1,1)
            nn.init.zeros_(self.layer.weight);nn.init.constant_(self.layer.bias,2.)
        else:self.layer=nn.Linear(384*5,nclasses)
    def forward(self,x):
        y=self.layer(x)
        return F.softplus(y) if self.kind=='depth' else y

def conf_update(cm,pred,target,n):
    valid=(target>=0)&(target<n)
    idx=(target[valid]*n+pred[valid]).long()
    cm+=torch.bincount(idx.cpu(),minlength=n*n).reshape(n,n)

def eval_probe(backbone,head,dl,kind,n,task):
    head.eval();backbone.eval()
    cm=torch.zeros((n,n),dtype=torch.int64)
    total=correct=0
    mse_sum=log_sum=log_sq_sum=count=0.
    for images, targets in dl:
        images=images.to(DEVICE,non_blocking=True); targets=targets.to(DEVICE,non_blocking=True)
        with torch.no_grad():
            f=feature_map(backbone,images,kind);p=head(f)
            if kind=='seg':
                p=F.interpolate(p,size=targets.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
                conf_update(cm,p,targets,n)
            elif kind=='cls':
                correct+=(p.argmax(1)==targets).sum().item();total+=targets.numel()
            else:
                p=F.interpolate(p.float(),size=targets.shape[-2:],mode='bilinear',align_corners=False)
                max_d=10.0
                valid=torch.isfinite(targets)&(targets>1e-3)&(targets<=max_d)
                if valid.any():
                    pp=p[valid].clamp(1e-3,max_d);gt=targets[valid]
                    e=pp-gt;mse_sum+=e.double().square().sum().item();count+=gt.numel()
                    diff=(pp.log()-gt.log()).double();log_sum+=diff.sum().item();log_sq_sum+=diff.square().sum().item()
    if kind=='seg':
        tp=cm.diag().double();union=cm.sum(0).double()+cm.sum(1).double()-tp;valid=union>0
        if not valid.any():raise RuntimeError('No labeled segmentation pixels!')
        iou=tp/union.clamp_min(1);return {'mIoU':iou[valid].mean().item(),'per_class_IoU':iou.tolist(),'n_images':len(dl.dataset)}
    if kind=='cls':return {'top1':correct/max(1,total),'n_images':len(dl.dataset)}
    if count==0:raise RuntimeError('No valid depth pixels! Check depth unit/scale and image masks.')
    v=log_sq_sum/count-(log_sum/count)**2
    return {'RMSE':math.sqrt(mse_sum/count),'SILog':100*math.sqrt(max(0,v)),'n_valid_pixels':int(count),'n_images':len(dl.dataset)}

def key_metric(kind,task):return 'mIoU' if kind=='seg' else 'top1' if kind=='cls' else 'RMSE'
def better(value,best,kind):return value<best if kind=='depth' else value>best

def head_loss(head,feat,targets,kind,task):
    pred=head(feat.detach())
    if kind=='cls':return F.cross_entropy(pred,targets)
    if kind=='seg':
        target_small=F.interpolate(targets[:,None].float(),size=pred.shape[-2:],mode='nearest')[:,0].long()
        # ADE excludes raw 0 which was mapped to 255; VOC ignores 255; Pet no ignore.
        return F.cross_entropy(pred,target_small,ignore_index=255)
    # depth: valid-weighted patch reduction (not bilinear interpolation of zeros)
    max_d=10.0
    valid=(torch.isfinite(targets)&(targets>1e-3)&(targets<=max_d)).float()
    values=torch.nan_to_num(targets,nan=0.,posinf=0.,neginf=0.)*valid
    den=F.interpolate(valid,size=pred.shape[-2:],mode='area')
    num=F.interpolate(values,size=pred.shape[-2:],mode='area')
    mask=den>.5
    if not mask.any():return pred.sum()*0.0
    gt=num/den.clamp_min(1e-5)
    return F.mse_loss(pred[mask],gt[mask])

def train_epochs(backbone,head,train_ds,kind,task,n,epochs, val_ds=None, name='', phase='selection', total_schedule_epochs=None):
    opt=torch.optim.AdamW(head.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=(total_schedule_epochs or epochs),eta_min=cfg.eta_min)
    dl=loader(train_ds,True,kind,task,seed_offset=7)
    vald=loader(val_ds,False,kind,task) if val_ds is not None else None
    best=math.inf if kind=='depth' else -math.inf
    best_ep=0;best_state=None;hist=[]
    for epoch in range(1,epochs+1):
        torch.manual_seed(cfg.seed+epoch) # identical augmentation RNG between models for each epoch
        head.train();lr=opt.param_groups[0]['lr'];total_loss=0;nb=0
        for x,y in dl:
            x=x.to(DEVICE,non_blocking=True);y=y.to(DEVICE,non_blocking=True)
            with torch.no_grad():f=feature_map(backbone,x,kind)
            opt.zero_grad(set_to_none=True);loss=head_loss(head,f,y,kind,task)
            loss.backward();opt.step();total_loss+=float(loss.item());nb+=1
        sch.step()
        record={'phase':phase,'epoch':epoch,'lr':lr,'train_loss':total_loss/max(1,nb)}
        if vald is not None:
            vals=eval_probe(backbone,head,vald,kind,n,task)
            record.update({'val_'+k:v for k,v in vals.items() if not isinstance(v,list)})
            val=vals[key_metric(kind,task)]
            if better(val,best,kind):
                best=val;best_ep=epoch
                best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        hist.append(record)
        score=record.get('val_'+key_metric(kind,task),'refit')
        print(f'[{name}] {phase} epoch {epoch}/{epochs} | loss={record["train_loss"]:.5f} | {score}')
    if vald is not None:
        if best_state is None:raise RuntimeError('Best validation state missing')
        head.load_state_dict(best_state)
    return hist,best_ep,best_state

def run_one(name,path,task):
    folder=OUT/name/task;folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json'
    if result_path.exists():
        print('SKIP completed',name,task)
        return json.loads(result_path.read_text())
    seed_all(cfg.seed)
    # Same task, same data object and official split for all models.
    if task not in _TASK_DATA_CACHE:
        _TASK_DATA_CACHE[task]=get_dataset(task)
    train_ds,val_ds,test_ds,nclasses,kind,info=_TASK_DATA_CACHE[task]
    assert len(train_ds)>0 and len(val_ds)>0
    model=build_backbone(path)
    seed_all(cfg.seed)
    head=Probe(kind,nclasses).to(DEVICE)
    epochs=cfg.seg_epochs if kind=='seg' else cfg.depth_epochs if kind=='depth' else cfg.cls_epochs
    start=time.time()
    try:
        hist,best_epoch,state=train_epochs(model,head,train_ds,kind,task,nclasses,epochs,val_ds,name)
        pd.DataFrame(hist).to_csv(folder/'selection_epochs.csv',index=False)
        metric=key_metric(kind,task)
        val_dl=loader(val_ds,False,kind,task)
        val=eval_probe(model,head,val_dl,kind,nclasses,task)
        if cfg.save_heads:torch.save(state,folder/'best_validation_probe.pt')
        row={'model':name,'task':task,'metric':metric,'best_epoch':best_epoch,'best_val':val[metric],
             'best_val_details':val,'dataset_protocol':info['split'],'dataset_source':info.get('data_source','original source'),'probe_kind':kind,'n_train_selection':len(train_ds),
             'n_val':len(val_ds),'n_test':len(test_ds) if test_ds is not None else None,
             'train_lr':cfg.lr,'min_lr':cfg.eta_min,'checkpoint':str(path),'selection_split':info['selection_split']}
        if test_ds is not None:
            # Test must only be evaluated after val selection.
            if cfg.refit_full_train:
                all_train=info.get('full_train')
                if all_train is None:raise RuntimeError('full_train dataset missing for heldout benchmark')
                seed_all(cfg.seed)
                refit_head=Probe(kind,nclasses).to(DEVICE)
                refit_hist,_,_=train_epochs(model,refit_head,all_train,kind,task,nclasses,best_epoch,None,name,phase='refit_full_train',total_schedule_epochs=epochs)
                pd.DataFrame(refit_hist).to_csv(folder/'refit_epochs.csv',index=False)
                if cfg.save_heads:torch.save({k:v.detach().cpu() for k,v in refit_head.state_dict().items()},folder/'refit_full_train_probe.pt')
                final_head=refit_head
                row['n_train_final']=len(all_train)
            else:
                final_head=head; row['n_train_final']=len(train_ds)
            heldout=eval_probe(model,final_head,loader(test_ds,False,kind,task),kind,nclasses,task)
            row['test_'+metric]=heldout[metric]
            row['test_details']=heldout
            if kind=='depth':row['test_SILog']=heldout['SILog']
        if kind=='depth':row['val_SILog']=val['SILog']
        row['elapsed_hours']=(time.time()-start)/3600
        result_path.write_text(json.dumps(row,indent=2,ensure_ascii=False,default=str))
        print('DONE',name,task,'best epoch',best_epoch,'val',val[metric],'test',row.get('test_'+metric))
        return row
    finally:
        del model,head;clear_cuda()



if __name__ == "__main__":
    _TASK_DATA_CACHE={}  # shared within this notebook kernel; never downloaded once per model
    models=dict(MODEL_PATHS)
    if PRETRAINED_REFERENCE:models['pretrained']=None
    results=[]
    for task in TASKS:  # task-first: load each benchmark only when necessary
        for name,path in models.items():
            result=run_one(name,path,task)
            results.append(result)
            # Save results even if future task fails.
            pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]).to_csv(OUT/'summary.csv',index=False)
            (OUT/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,default=str))
    print('Saved',OUT/'summary.csv')

    import pandas as pd, shutil
    p=OUT/'summary.csv'
    if p.exists():
        results=pd.read_csv(p)
        display(results)
    else:print('Run the evaluation runner first')

    # Save per-task validation curves and selected epochs for report figures.
    import matplotlib.pyplot as plt
    for name in models:
        for task in TASKS:
            d=OUT/name/task
            f=d/'selection_epochs.csv'
            if not f.exists(): continue
            df=pd.read_csv(f)
            valcols=[k for k in ['val_mIoU','val_RMSE','val_top1'] if k in df.columns]
            if not valcols: continue
            metric=valcols[0]
            fig,ax=plt.subplots(figsize=(7.2,4.1))
            ax.plot(df['epoch'],df[metric],marker='o',markersize=3,lw=1.5)
            best_i=df[metric].idxmin() if metric=='val_RMSE' else df[metric].idxmax()
            ax.scatter([df.loc[best_i,'epoch']],[df.loc[best_i,metric]],s=80,marker='*',zorder=3)
            ax.set(xlabel='Probe epoch',ylabel=metric,title=f'{name} — {task}: validation {metric}')
            ax.grid(alpha=.2);fig.tight_layout();fig.savefig(d/'validation_curve.png',dpi=160);plt.close(fig)

    # Package metrics and heads. Large .pt model inputs are not copied into the zip.
    archive=shutil.make_archive(str(OUT)+'_results','zip',root_dir=OUT)
    print('Archive:',archive)
    try:
        from IPython.display import FileLink, display
        display(FileLink(archive))
    except Exception:pass
