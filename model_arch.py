import torch
import torchvision.models as models
import torch.nn as nn
import torchvision
import torchvision.models.detection as det_model
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import AnchorGenerator
from torchvision import transforms as trf
import random

class FuseStyle(nn.Module):
    """FuseStyle.
    Reference:
      Zhou et al. Domain Generalization with FuseStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using FuseStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'FuseStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        x.permute(*torch.arange(x.ndim-1,-1,-1))
        y=x_normed.view(B,-1)
        #n = torch.norm(y,dim=1)
        #n = torch.tensordot(n,n.T,dims=0)
        z = torch.tensordot(y,y.mT,dims=1)
        perm = torch.argmin(z, dim=0)
                
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        z = x_normed*sig_mix + mu_mix
        return z
        

def load_backbone():
    """A torch vision retinanet model"""
    #BB = torchvision.models.detection.retinanet_resnet50_fpn(weights=det_model.RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    BB = torchvision.models.detection.retinanet_resnet50_fpn(weights=det_model.RetinaNet_ResNet50_FPN_Weights.COCO_V1)
    return BB


def create_anchor_generator():
                            
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [16, 32, 64,128,256])
    aspect_ratios = ((1.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    return anchor_generator



def create_model(num_classes,layers,p=0.5,alpha=0.3):

    
    #resnet = load_backbone()
    #backbone = resnet.backbone
    fs = FuseStyle(p=p,alpha=alpha)
    anchor_generator = create_anchor_generator()
    model = det_model.retinanet_resnet50_fpn(trainable_backbone_layers=5,
                                             weights="DEFAULT")
    #model.anchor_generator = anchor_generator
    model.head.classification_head.cls_logits.out_channels=18 # 9 * num_classes
    model.head.regression_head.bbox_reg.out_channels=36
    if 'layer1' in layers:
        #backbone.body.layer1.add_module('FuseStyle',fs)
        model.backbone.body.layer1.add_module('FuseStyle',fs)
    if 'layer2' in layers:
        #backbone.body.layer2.add_module('FuseStyle',fs)
        model.backbone.body.layer2.add_module('FuseStyle',fs)
    if 'layer3' in layers:
        #backbone.body.layer3.add_module('FuseStyle',fs)
        model.backbone.body.layer3.add_module('FuseStyle',fs)
    if 'layer4' in layers:
        #backbone.body.layer4.add_module('FuseStyle',fs)
        model.backbone.body.layer4.add_module('FuseStyle',fs)
    #model = RetinaNet(backbone=backbone, num_classes=num_classes, anchor_generator=anchor_generator)
    #model.transform = trf.Resize((800,), max_size=1333,interpolation=trf.InterpolationMode.BILINEAR)
    #model.transform.image_mean=[0.7761,0.5452,0.7296]
    #model.transform.image_std=[0.1455,0.1933,0.1351]
    #model.transform.image_mean=[0,0,0]
    #model.transform.image_std=[1,1,1]          
    return model
