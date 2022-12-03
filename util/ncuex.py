import torch
import numpy as np
import torchvision.transforms as transforms
from torchray.attribution.gradient import gradient

class ncuex(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, net, p=0.5,length=8, arng=[8,18], rescle =False):
        self.p = p
        self.net = net
        self.length = length
        self.rescle = rescle
        self.arng = arng

    def __call__(self, img):
        
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        '''
    p: probability of gradient happening
    '''
        if np.random.uniform(0, 1) > self.p:
            return img.cuda()

        self.net.eval()
        imgc = img.unsqueeze(0).cuda()
        pred = self.net(imgc)
        _, l = torch.max(pred.data, 1)
        saliency = gradient(self.net, imgc, l.item())
        x = saliency.squeeze()
        xloc,yloc = np.unravel_index(x.cpu().argmax(), (32,32))
        mask = np.ones((32, 32), np.float32)
        if self.rescle == True:

            if self.length ==0:
                lengt = np.random.randint(self.arng[0], self.arng[1])
                y1 = np.clip(yloc - lengt // 2, 0, 32)
                y2 = np.clip(yloc + lengt // 2, 0, 32)
                x1 = np.clip(xloc - lengt // 2, 0, 32)
                x2 = np.clip(xloc + lengt // 2, 0, 32)
                mask[x1: x2, y1: y2] = 0.
                mask = torch.from_numpy(mask)
                mask = mask.expand_as(imgc)
                img = imgc * mask.cuda()
                b = imgc*(1/(1-(lengt**2/32**2)))
                self.net.train()
                return b.squeeze(0)
            else:
                y1 = np.clip(yloc - self.length // 2, 0, 32)
                y2 = np.clip(yloc + self.length // 2, 0, 32)
                x1 = np.clip(xloc - self.length // 2, 0, 32)
                x2 = np.clip(xloc + self.length // 2, 0, 32)
                mask[x1: x2, y1: y2] = 0.
                mask = torch.from_numpy(mask)
                mask = mask.expand_as(imgc)
                img = imgc * mask.cuda()
                b = imgc*(1/(1-(self.length**2/32**2)))
                self.net.train()
                return b.squeeze(0)

        else:

            if self.length ==0:
                lengt = np.random.randint(8, 18)
                y1 = np.clip(yloc - lengt // 2, 0, 32)
                y2 = np.clip(yloc + lengt // 2, 0, 32)
                x1 = np.clip(xloc - lengt // 2, 0, 32)
                x2 = np.clip(xloc + lengt // 2, 0, 32)
                mask[x1: x2, y1: y2] = 0.
                mask = torch.from_numpy(mask)
                mask = mask.expand_as(imgc)
                img = imgc * mask.cuda()
                b = imgc
                self.net.train()
                return b.squeeze(0)
            else:
                y1 = np.clip(yloc - self.length // 2, 0, 32)
                y2 = np.clip(yloc + self.length // 2, 0, 32)
                x1 = np.clip(xloc - self.length // 2, 0, 32)
                x2 = np.clip(xloc + self.length // 2, 0, 32)
                mask[x1: x2, y1: y2] = 0.
                mask = torch.from_numpy(mask)
                mask = mask.expand_as(imgc)
                img = imgc * mask.cuda()
                b = imgc
                self.net.train()
                return b.squeeze(0)       


