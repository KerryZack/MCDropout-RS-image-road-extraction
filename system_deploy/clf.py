import torch
from PIL import Image
import time
from models import DLinkNet
#import cv2
import numpy as np
import torch.nn as nn
import monai
from monai.data import  PILReader
from monai.transforms import *

def predict(image_path):
    # if option =="resnet101":
    #     model = models.resnet101(pretrained=True)
    # elif option =="resnet50":
    #     model = models.resnet50(pretrained=True)
    # elif option == "densenet121":
    #     model = models.densenet121(pretrained=True)
    # elif option == "shufflenet_v2_x0_5":
    #     model = models.shufflenet_v2_x0_5(pretrained=True)
    # else:
    #     model = models.mobilenet_v2(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建Uncertainty network并加载权重
    # pretrained_dict = torch.load('./checkpoint/checkpoint_chasefcn67.pth' ,map_location=device)
    # fcn = tiramisu.FCDenseNet67(n_classes=1).to(device)
    # model_dict = fcn.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # fcn.load_state_dict(model_dict)

    # # 创建Segmentation network并加载权重
    # sg_pretrained_dict = torch.load("./checkpoint/checkpoint_unetmean.pth",map_location=device )
    # Sg_model = stdunet.build_unet(n_channels=3, n_classes=1).to(device)
    # # sgmodel_dict = Sg_model.state_dict()
    # # sg_pretrained_dict = {k: v for k, v in sg_pretrained_dict.items() if k in sgmodel_dict}
    # # sgmodel_dict.update(sg_pretrained_dict)
    # Sg_model.load_state_dict(sg_pretrained_dict['model_state_dict'])


    model = DLinkNet.DLinkNet34(num_classes= 2, drop_rate=0)
    model = torch.nn.DataParallel(model.cuda(), device_ids=[0])

    if torch.cuda.is_available():
        model = model.to(device)
    # print('==> Loading %s model file: %s' %
    #       (model.__class__.__name__, model_file))
    checkpoint = torch.load("./checkpoint/best.pth.tar")
    model.load_state_dict(checkpoint['model_state_dict'])

    
    
    # image = Image.open(image_path)
    # print(image)
    # print(image_path)
    
    transforms_monai = Compose([
    # LoadImage(image_only=True, reader=PILReader, dtype=np.float32),  # 直接加载单张图像
    # EnsureChannelFirst(),  # 将通道移动到第一个维度
    SpatialPad(spatial_size=1504),  # 填充到指定大小
    EnsureType(),  # 确保输出是 torch.Tensor
    ])
    
    
    # # 使用 PIL 加载图片
    # pil_image = Image.open(image_path)  # 加载图像

    # 将 PIL 图像转换为 NumPy 数组
    image_array = np.array(image_path, dtype=np.float32)

    # 将 NumPy 数组转换为 PyTorch Tensor
    image_tensor = torch.tensor(image_array)

    # 如果是灰度图像（单通道），添加一个通道维度
    if len(image_tensor.shape) == 2:  # 如果形状是 [H, W]
        image_tensor = image_tensor.unsqueeze(0)  # 添加通道维度，变成 [C, H, W]

    # 如果是彩色图像（RGB），将通道移动到第一个维度
    elif len(image_tensor.shape) == 3 and image_tensor.shape[-1] == 3:  # 如果形状是 [H, W, C]
        image_tensor = image_tensor.permute(2, 0, 1)  # 变成 [C, H, W]

    # # 应用剩余的 MONAI 变换
    image = transforms_monai(image_tensor)
    

    # image = transforms_monai(image_path)
    
    image = torch.unsqueeze(image, 0)
    # print(image.shape)
    
    image = image.to(device)
    # print(image.shape)

    model.eval()
    results = []
    for i in range(10):
        prediction,_ = model(image)
        prediction = torch.softmax(prediction, dim=1)
        results.append(prediction)
    results = torch.stack(results, dim=0) # mc_samples,batch_size,num_classes,h,w

    pred = torch.mean(results, dim=0) # batch_size,num_classes,h,w
    pred = torch.argmax(pred, dim=1) # batch_size,h,w
    pred = pred.squeeze()
    pred = pred.cpu().detach().numpy()
    # mask = np.where(pred>0.5,0,1)

    return pred

    # with open('imagenet_classes.txt') as f:
    #     classes = [line.strip() for line in f.readlines()]
    # prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    # _, indices = torch.sort(out, descending=True)

    # return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]],fps


