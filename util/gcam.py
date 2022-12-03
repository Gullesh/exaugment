import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def gcam(net,a,l, target_layers, p=0.5,intval=0.8, sal=False):
    '''
    a : batch of images
    p: probability of gradient happening
    '''
    net.eval()
    bsize = a.shape[0]
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(m.item()) for m in l]
    grayscale_cam = cam(input_tensor=a, targets=targets)
    b = torch.from_numpy(grayscale_cam)
    saliency = b.unsqueeze(1).cuda()
    AA = saliency.clone()
    AA = AA.view(saliency.size(0), -1)
    con = torch.quantile(AA, intval, interpolation='nearest', dim=1)
    if sal:
        AA =(AA>con.reshape(bsize,1)).float()
    else:
        AA =(AA<con.reshape(bsize,1)).float()
    masks = AA.view(bsize, 1, 32, 32)
    AAA = masks.expand_as(a)
    fo = a*AAA
    rnd = ((torch.rand(size=(bsize,1,1,1)) < p).int()).cuda()
    foo = rnd.cuda()*fo
    aa = ((1-rnd)*a) + foo
    net.train()
    return aa