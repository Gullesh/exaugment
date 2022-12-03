import torch
from torchray.attribution.gradient import gradient
import numpy as np

def cuex(net,a,l, p=0.5,length=8, rescle =False,arng=[8,18]):
    '''
    a : batch of images
    p: probability of gradient happening
    '''
    bsize = a.shape[0]
    new = torch.empty((1,3,32,32)).cuda()
    net.eval()
    if rescle == True:
            
        for i in range(bsize):
            if np.random.uniform(0, 1) > p:
                b = a[i].unsqueeze(0)
                new = torch.cat((new,b),dim = 0)
            else:
                saliency = gradient(net, a[i].unsqueeze(0), l[i])
                x = saliency.squeeze()
                xloc,yloc = np.unravel_index(x.cpu().argmax(), (32,32))
                mask = np.ones((32, 32), np.float32)
                if length ==0:
                    lengt = np.random.randint(arng[0], arng[1])
                    y1 = np.clip(yloc - lengt // 2, 0, 32)
                    y2 = np.clip(yloc + lengt // 2, 0, 32)
                    x1 = np.clip(xloc - lengt // 2, 0, 32)
                    x2 = np.clip(xloc + lengt // 2, 0, 32)
                    mask[x1: x2, y1: y2] = 0.
                    mask = torch.from_numpy(mask)
                    mask = mask.expand_as(a[i])
                    img = a[i] * mask.cuda()
                    b = img.unsqueeze(0)*(1/(1-(lengt**2/32**2)))
                    new = torch.cat((new,b),dim = 0)
                else:
                    y1 = np.clip(yloc - length // 2, 0, 32)
                    y2 = np.clip(yloc + length // 2, 0, 32)
                    x1 = np.clip(xloc - length // 2, 0, 32)
                    x2 = np.clip(xloc + length // 2, 0, 32)
                    mask[x1: x2, y1: y2] = 0.
                    mask = torch.from_numpy(mask)
                    mask = mask.expand_as(a[i])
                    img = a[i] * mask.cuda()
                    b = img.unsqueeze(0)*(1/(1-(length**2/32**2)))
                    new = torch.cat((new,b),dim = 0)

        net.train()	
        return new[1:(bsize+1)]
    else:
                    
        for i in range(bsize):
            if np.random.uniform(0, 1) > p:
                b = a[i].unsqueeze(0)
                new = torch.cat((new,b),dim = 0)
            else:
                saliency = gradient(net, a[i].unsqueeze(0), l[i])
                x = saliency.squeeze()
                xloc,yloc = np.unravel_index(x.cpu().argmax(), (32,32))
                mask = np.ones((32, 32), np.float32)
                if length ==0:
                    lengt = np.random.randint(8, 20)
                    y1 = np.clip(yloc - lengt // 2, 0, 32)
                    y2 = np.clip(yloc + lengt // 2, 0, 32)
                    x1 = np.clip(xloc - lengt // 2, 0, 32)
                    x2 = np.clip(xloc + lengt // 2, 0, 32)
                    mask[x1: x2, y1: y2] = 0.
                    mask = torch.from_numpy(mask)
                    mask = mask.expand_as(a[i])
                    img = a[i] * mask.cuda()
                    b = img.unsqueeze(0)
                    new = torch.cat((new,b),dim = 0)

                else:
                    y1 = np.clip(yloc - length // 2, 0, 32)
                    y2 = np.clip(yloc + length // 2, 0, 32)
                    x1 = np.clip(xloc - length // 2, 0, 32)
                    x2 = np.clip(xloc + length // 2, 0, 32)
                    mask[x1: x2, y1: y2] = 0.
                    mask = torch.from_numpy(mask)
                    mask = mask.expand_as(a[i])
                    img = a[i] * mask.cuda()
                    b = img.unsqueeze(0)
                    new = torch.cat((new,b),dim = 0)

        net.train()	
        return new[1:(bsize+1)]
