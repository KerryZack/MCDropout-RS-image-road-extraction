from ctypes import Union
from datetime import datetime
import os
import os.path as osp
import numpy as np 
import random
# PyTorch includes
import torch
from torchvision import transforms
from torch.utils import model_zoo
from torch.utils.data import DataLoader
import argparse
import yaml
import torch.nn
# Custom includes
from utils import Trainer
import monai
from monai.data import decollate_batch, PILReader
from monai.transforms import *
# import segmentation_models_pytorch as smp
from model import DeepLabV3Plus
from utils.palette import CLASSES,PALETTE
from utils.data_manager import DeepGlobe

here = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
parser.add_argument('--gpus', type=list, default=[0], help='gpu id')
parser.add_argument('--resume', type=str, default='logs/deeplabv3plus/20230519_012505.068336/best.pth.tar', help='checkpoint path')
parser.add_argument(
        '--datasetdir', type=str, default='../../dataset', help='The address of the dataset'
    )
parser.add_argument(
        '--num-classes',type=int,default=2,help='number of classes'
)
parser.add_argument(
        '--batch-size', type=int, default=2, help='batch size for training the model'
    )
parser.add_argument(
        '--num-workers',type=int,default=4,help='how many subprocesses to use for dataloading.'
)
parser.add_argument(
        '--input-size',type=int,default=256,help='input image size'
    )
parser.add_argument(
        '--dropout-rate',type=float,default=0.1,help='mc dropout rate'
    )
parser.add_argument(
        '--mc-samples', type=int, default=10, help='mc dropout sample times'
    )
parser.add_argument(
        '--lambda-kd', type=float, default=0.5, help='mc dropout sample times'
    )
parser.add_argument(
        '--lambda-soft', type=float, default=0.5, help='mc dropout sample times'
    )
parser.add_argument(
        '--max-epoch', type=int, default=100, help='max epoch'
    )
parser.add_argument(
        '--stop-epoch', type=int, default=100, help='stop epoch'
    )
parser.add_argument(
        '--interval-validate', type=int, default=1, help='interval epoch number to valide the model'
    )
parser.add_argument(
        '--lr-model', type=float, default=1e-4, help='learning rate'
    )
parser.add_argument(
        '--seed',type=int,default=1234,help='set random seed'
    )
parser.add_argument(
        '--model-name',type=str,default='deeplabv3plus',help='network name. \
        Options:unet,unetplusplus,manet,linknet,fpn,pspnet,deeplabv3,deeplabv3plus,pan'
    )
parser.add_argument(
        "--restore-from", type=str, default='http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth',
        help="Where restore model parameters from.")
args = parser.parse_args()

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def main():
#The part of init
    now = datetime.now()
    args.out = osp.join(here, 'logs', now.strftime(osp.join(args.model_name,'S_%Y%m%d_%H%M%S.%f')))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    cuda = torch.cuda.is_available()
    torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

#The part of dataset
    dataset = DeepGlobe(root = args.datasetdir)
    train_files, val_files = dataset.train_files, dataset.val_files

# 1. dataset
    train_transforms = transforms.Compose([
            LoadImaged(
                keys=["img", "label"], reader=PILReader, dtype=np.float32
            ),  # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["img"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["img"], allow_missing_keys=True
            ),  # Do not scale label
            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=args.input_size, random_size=False
            ),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img", "label"],
                prob=0.15,
                min_zoom=0.8,
                max_zoom=1.5,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["img", "label"]),
            SqueezeDimd(keys=["label"],dim=0)
    ])

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.float32),
            #AddChanneld(keys=["label"], allow_missing_keys=True), #if augmented label， release this code.
            AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),   #将通道移动到第一个维度
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),   #将图像的像素值线性缩放到 [0, 1] 范围。
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label"]),
            #SqueezeDimd(keys=["label"],dim=0        
        ]
    )

#  create a training data loader
    train_dataset = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    # create a validation data loader
    val_dataset = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
    val_dataset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

# 2. model  
    t_model=DeepLabV3Plus(encoder_weights="imagenet",classes=args.num_classes,dropout_rate=args.dropout_rate)
    t_model=torch.nn.DataParallel(t_model.cuda(),device_ids=args.gpus)

    s_model = DeepLabV3Plus(encoder_weights="imagenet",classes=args.num_classes,dropout_rate=0)
    s_model=torch.nn.DataParallel(s_model.cuda(),device_ids=args.gpus)

# 3. optimizer
    optim_model = torch.optim.Adam(
        s_model.parameters(),
        lr=args.lr_model,
        betas=(0.9, 0.99)
    )

# 4. resume
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = t_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        t_model.load_state_dict(model_dict)
        # start_epoch = checkpoint['epoch'] + 1
        # start_iteration = checkpoint['iteration'] + 1
        # optim_model.load_state_dict(checkpoint['optim_state_dict'])
        t_model.eval()
        enable_dropout(t_model)

#5. Trainer
    trainer = Trainer.Trainer(
        cuda=cuda,
        t_model=t_model,
        s_model=s_model,
        optimizer_model=optim_model,
        lr_gen=args.lr_model,
        mc_samples=args.mc_samples,
        lambda_kd = args.lambda_kd,
        lambda_soft=args.lambda_soft,
        loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        warmup_epoch=-1,
        classes=CLASSES,
        palette=PALETTE
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()
