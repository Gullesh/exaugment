import torch
import numpy as np
import torchvision.transforms as transforms

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,p=.25,rescle=False ):
        self.p = p
        self.rescle = rescle
        self.length = np.random.randint(16,22)

    def __call__(self, img):
        
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if np.random.uniform(0, 1) > self.p:
            return img
        else:
            h = img.size(1)
            w = img.size(2)

            mask = np.zeros((h, w), np.float32)

        
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 1.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
            if self.rescle:
                img = img*(1/(1-(self.length**2/32**2)))
            return img
