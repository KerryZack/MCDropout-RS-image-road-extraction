import os.path as osp
import numpy as np
import os
from .convertrgb import out_to_rgb_np
from .palette import CLASSES,PALETTE
from torchvision import transforms
from utils.metrics import *
from PIL import Image



def saveimages(path,imagelist,label_list,pred_list,imagename_list):
    tran=transforms.ToPILImage()
    if not osp.exists(path):
        os.makedirs(path)    
    if not osp.exists(osp.join(path,'raw/')):
        os.makedirs(osp.join(path,'raw/'))
    if not osp.exists(osp.join(path,'label/')):
        os.makedirs(osp.join(path,'label/'))    
    if not osp.exists(osp.join(path,'prediction/')):
        os.makedirs(osp.join(path,'prediction/'))

    for i in range(len(imagelist)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        label=label_list[i]
        save_label=out_to_rgb_np(torch.squeeze(label.cpu().detach()),PALETTE,CLASSES)
        save_label=save_label.transpose(1,0,2)
        label=tran(save_label)
        label.save(osp.join(osp.join(path,'label/'),tmpimage_name))

        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        pred=pred_list[i]
        save_pred=out_to_rgb_np(pred.cpu().detach(),PALETTE,CLASSES)
        save_pred=save_pred.transpose(1,0,2)
        prediction=tran(save_pred)  
        prediction.save(osp.join(osp.join(path,'prediction/'),tmpimage_name))

        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        image=imagelist[i]
        image=image.permute(0,2,1)
        image=tran(image)
        #PIL 分割结果叠加图像显示
        image=image.convert('RGBA')
        prediction=prediction.convert('RGBA')
        image = Image.blend(image, prediction, 0.5)
        tmpimage_name = osp.splitext(tmpimage_name)[0] + '.png'

        image.save(osp.join(osp.join(path,'raw/'),tmpimage_name))


def saveentropy(path, type, img_list, imagename_list):
    tran=transforms.ToPILImage()
    if not osp.exists(path):
        os.makedirs(path)
    if not osp.exists(osp.join(path, type)):
        os.makedirs(osp.join(path, type))

    for i in range(len(img_list)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        img = img_list[i]
        # save_img=out_to_rgb_np(torch.squeeze(img.cpu().detach()),PALETTE,CLASSES)
        save_img=torch.squeeze(img.cpu().detach())
        # save_img=save_img.transpose(1,0,2)
        label=tran(save_img)
        label=label.rotate(-90)
        label=label.transpose(Image.FLIP_LEFT_RIGHT)

        label.save(osp.join(osp.join(path,type),tmpimage_name))


def savepesudo(path, type, img_list, imagename_list):
    tran=transforms.ToPILImage()
    if not osp.exists(path):
        os.makedirs(path)
    if not osp.exists(osp.join(path, type)):
        os.makedirs(osp.join(path, type))

    for i in range(len(img_list)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        img = img_list[i]
        # save_img=out_to_rgb_np(torch.squeeze(img.cpu().detach()),PALETTE,CLASSES)
        save_img=torch.squeeze(img.cpu().detach())
        # save_img=save_img.transpose(1,0,2)
        label=tran(save_img)
        label=label.rotate(-90)
        label=label.transpose(Image.FLIP_LEFT_RIGHT)

        label.save(osp.join(osp.join(path,type),tmpimage_name))


def sigmoid_rampup(current,max_epoch):
    if max_epoch == 0:
        return 1.0
    else:
        current = np.clip(current,0.0,max_epoch)
        phase = 1.0 - current/max_epoch
        return float(np.exp(-5.0 * phase * phase))