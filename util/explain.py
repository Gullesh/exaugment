import torch
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from cifar10_models.resnet import resnet18, resnet34, resnet50


class Explain(object):
    """masking salient regions out of original images.
    Args:
        probability: probability of doing this data augmentation happen.
    """

    def __init__(self, cnn, probability=0.5):
        self.probability = probability
        self.cnn = cnn

    def __call__(self, img):

        if np.random.uniform(0, 1) > self.probability:
            return img.cuda()

        """
                Args:
                    img (Tensor): Tensor image of size (C, H, W).
                Returns:
                    Tensor: perturbated image with salient region masked.
        """
        # Hyper parameters.
        tv_beta = 3
        learning_rate = 0.1
        max_iterations = 100
        l1_coeff = 0.01
        tv_coeff = 0.2
        use_cuda = True
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        Tensor = FloatTensor

        # should load the exact same model somehow
        # Pretrained model
        self.cnn.eval()

        original_img = img.numpy()
        original_img = cv2.normalize(original_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        original_img.astype(np.uint8)
        original_img = np.transpose(original_img, (1, 2, 0))
        imeg = np.float32(original_img) / 255

        blurred_img2 = np.float32(cv2.medianBlur(original_img, 5)) / 255
        mask_init = np.ones((28, 28), dtype=np.float32)

        # preprocessing
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2471, 0.2435, 0.2616]
        preprocessed_img = imeg.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

        if use_cuda:
            preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
        else:
            preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

        preprocessed_img_tensor.unsqueeze_(0)
        imeg = Variable(preprocessed_img_tensor, requires_grad=False)

        preprocessed_img2 = blurred_img2.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img2[:, :, i] = preprocessed_img2[:, :, i] - means[i]
            preprocessed_img2[:, :, i] = preprocessed_img2[:, :, i] / stds[i]
        preprocessed_img2 = \
            np.ascontiguousarray(np.transpose(preprocessed_img2, (2, 0, 1)))

        if use_cuda:
            preprocessed_img_tensor2 = torch.from_numpy(preprocessed_img2).cuda()
        else:
            preprocessed_img_tensor2 = torch.from_numpy(preprocessed_img2)

        preprocessed_img_tensor2.unsqueeze_(0)
        blurred_img = Variable(preprocessed_img_tensor2, requires_grad=False)

        #numpy to torch
        output = np.float32([mask_init])
        output = torch.from_numpy(output)
        if use_cuda:
            output = output.cuda()
        output.unsqueeze_(0)
        mask = Variable(output, requires_grad=True)


        if use_cuda:
            upsample = torch.nn.UpsamplingBilinear2d(size=(32, 32)).cuda()
        else:
            upsample = torch.nn.UpsamplingBilinear2d(size=(32, 32))
        optimizer = torch.optim.Adam([mask], lr=learning_rate)

        target = torch.nn.Softmax(dim=1)(self.cnn(imeg))
        category = np.argmax(target.cpu().data.numpy())
        # for i in range(max_iterations):
        for i in range(50):
            upsampled_mask = upsample(mask)
            # The single channel mask is used with an RGB image,
            # so the mask is duplicated to have 3 channel,
            upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))
            # Use the mask to perturbated the input image.
            perturbated_input = imeg.mul(upsampled_mask) + blurred_img.mul(1 - upsampled_mask)
            # noise = np.zeros((32, 32, 3), dtype=np.float32)
            # cv2.randn(noise, 0, 0.2)
            # noise = numpy_to_torch(noise)
            # perturbated_input = perturbated_input + noise
            tvinput = mask[0, 0, :]
            row_grad = torch.mean(torch.abs((tvinput[:-1, :] - tvinput[1:, :])).pow(tv_beta))
            col_grad = torch.mean(torch.abs((tvinput[:, :-1] - tvinput[:, 1:])).pow(tv_beta))
            tv_norm = row_grad + col_grad
            outputs = torch.nn.Softmax(dim=1)(self.cnn(perturbated_input))
            loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + tv_coeff * tv_norm+ outputs[0, category]
            # = outputs[0, category]
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Optional: clamping seems to give better results
            mask.data.clamp_(0, 1)
        return perturbated_input[0]
