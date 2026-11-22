"""Frozen-backbone probe components used in the final evaluation notebook.

The full benchmark dataset setup and validation/test split policy are kept in
AGA_Full_7_Benchmarks_Frozen_Probes.ipynb; those steps are not duplicated here.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

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
                max_d=80.0 if task=='kitti' else 10.0
                valid=torch.isfinite(targets)&(targets>1e-3)&(targets<=max_d)
                if task=='kitti':
                    # Garg/Eigen crop applied to resized target, evaluated in consistent image-relative coordinates.
                    _,_,h,w=targets.shape
                    crop=torch.zeros_like(valid)
                    crop[:,:,int(.40810811*h):int(.99189189*h),int(.03594771*w):int(.96405229*w)]=True
                    valid &=crop
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
    max_d=80.0 if task=='kitti' else 10.0
    valid=(torch.isfinite(targets)&(targets>1e-3)&(targets<=max_d)).float()
    values=torch.nan_to_num(targets,nan=0.,posinf=0.,neginf=0.)*valid
    den=F.interpolate(valid,size=pred.shape[-2:],mode='area')
    num=F.interpolate(values,size=pred.shape[-2:],mode='area')
    mask=den>.5
    if not mask.any():return pred.sum()*0.0
    gt=num/den.clamp_min(1e-5)
    return F.mse_loss(pred[mask],gt[mask])
