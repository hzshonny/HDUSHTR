import paddle
from paddle import nn
import paddle.nn.functional as F
from tensorboardX import SummaryWriter
from models.discriminator import Discriminator_STE
from paddle import autograd
from PIL import Image
import numpy as np

def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = paddle.bmm(feat, feat_t) / (ch * h * w)
    return gram

def visual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

def dice_loss(input, target):
    input = F.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    
    input = input 
    target = target

    a = paddle.sum(input * target, 1)
    b = paddle.sum(input * input, 1) + 0.001
    c = paddle.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = paddle.mean(d)
    return 1 - dice_loss

def bce_loss(input, target):
    input = F.sigmoid(input)

    input = input.reshape([input.shape[0], -1])
    target = target.reshape([target.shape[0], -1])
    
    input = input 
    target = target

    bce = paddle.nn.BCELoss()
    
    return bce(input, target)

def dice_bce_loss(input, target):
    input = F.sigmoid(input)
    input_flat = paddle.reshape(input,(input.shape[0], -1))
    target_flat = paddle.reshape(target,(target.shape[0], -1))

    a = sum(input_flat * target_flat, 1)
    b = sum(input_flat * input_flat, 1) + 0.001
    c = sum(target_flat * target_flat, 1) + 0.001
    dice_l = paddle.mean(2 * a / (b + c))

    bce_l = F.binary_cross_entropy_with_logits(input_flat, target_flat)

    #加权求和
    weight_dice = 0.5
    loss = weight_dice * dice_l + (1 - weight_dice) * bce_l

    return loss

class LossWithGAN_STE(nn.Layer):
    # def __init__(self, logPath, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
    def __init__(self,logPath,extractor,Lamda, lr, betasInit=(0.5, 0.9)):
        super(LossWithGAN_STE, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.discriminator = Discriminator_STE(3)    ## local_global sn patch gan
        self.D_optimizer = paddle.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = paddle .cuda.is_available()
        self.numOfGPUs = paddle.cuda.device_count()
        self.lamda = Lamda
        self.writer = SummaryWriter(logPath)

    def forward(self, input, mask, x_o1,x_o2,x_o3,output,mm, gt, count, epoch):
        self.discriminator.zero_grad()
        D_real = self.discriminator(gt, mask)
        D_real = D_real.mean().sum() * -1
        D_fake = self.discriminator(output, mask)
        D_fake = D_fake.mean().sum() * 1
        D_loss = paddle.mean(F.relu(1.+D_real)) + paddle.mean(F.relu(1.+D_fake))
        D_fake = -paddle.mean(D_fake)

        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        self.D_optimizer.step()

        holeLoss =  self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = self.l1(mask * output, mask * gt)  
        mask_loss = dice_bce_loss(mm, 1-mask)

        self.writer.add_scalar('Hole loss',holeLoss.item(),count)
        self.writer.add_scalar('Valid loss', validAreaLoss.item(), count)
        self.writer.add_scalar('Mask loss', mask_loss.item(), count)
        self.writer.add_scalar('G loss', D_fake.item(), count)
        GLoss =  mask_loss + holeLoss + validAreaLoss + 0.05*D_fake
        self.writer.add_scalar('Generator/Joint loss', GLoss.item(), count)
        return GLoss.sum()
    
