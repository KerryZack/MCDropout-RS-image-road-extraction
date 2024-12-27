from datetime import datetime
import os
import os.path as osp
import timeit
import numpy as np
import pytz
import torch
import torch.nn.functional as F
# from myloss import myloss
from tensorboardX import SummaryWriter
import tqdm
import socket
from utils.metrics import SegmentationMetric
from utils.Utils import *
from utils.losses import *
from .convertrgb import out_to_rgb
import matplotlib.pyplot as plt
from utils.ramps import *


'''loss fuction'''
bceloss = torch.nn.BCELoss()  # You should for output to use sigmoid
mseloss = torch.nn.MSELoss()
celoss = torch.nn.CrossEntropyLoss()
import segmentation_models_pytorch as smp

diceloss = smp.losses.DiceLoss(mode='multiclass')
kd_loss = kd_loss()
kl_loss = torch.nn.KLDivLoss(log_target=True,reduction="batchmean")

# 获取学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 更新教师模型参数
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        # ema_param.data.mul_(alpha).add_(param.data, * , 1 - alpha)


def get_current_consistency_weight(self, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)


class Trainer(object):

    def __init__(self, cuda, t_model, s_model, optimizer_model,
                 loader, val_loader, out, max_epoch, classes, palette, stop_epoch=None, lr_gen=1e-3,
                 mc_samples=10, lambda_kd=0, lambda_soft=0,
                 interval_validate=None, batch_size=8, warmup_epoch=-1, ema_decay=0.99,
                 consistency=0.1, consistency_rampup=200, together = True ):
        self.cuda = cuda
        self.t_model = t_model
        self.s_model = s_model
        self.warmup_epoch = warmup_epoch
        self.optim_model = optimizer_model
        self.lr_gen = lr_gen
        self.mc_samples = mc_samples
        self.lambda_kd = lambda_kd
        self.lambda_soft = lambda_soft
        self.batch_size = batch_size
        self.loader = loader
        self.val_loader = val_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))
        self.ema_decay = ema_decay
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.consistency_criterion = softmax_dice_loss
        self.together = together

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'valid/loss_CE',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iter_num = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch

        self.best_mean_IoU = 0.0
        self.best_epoch = -1
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, 100, eta_min=1e-6,
        #                                                                last_epoch=-1)
        self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer_model, 100, power=0.9,verbose=True)
        self.palette = palette
        self.classes = classes

    def validate(self):
        # torch.cuda.empty_cache()
        training = self.s_model.training
        self.s_model.eval()
        seg = SegmentationMetric(len(self.classes))
        val_ce_loss = 0
        val_de_loss = 0
        metrics = []
        with torch.no_grad():
            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['img']
                label = sample['label'].long()

                if self.cuda:
                    data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    predictions,_ = self.s_model(data)

                loss_ce = ce_loss(predictions, label)
                loss_de = dice_loss(predictions, label.unsqueeze(1).cuda())

                loss_data_ce = loss_ce.data.item()
                loss_data_de = loss_de.data.item()
                if np.isnan(loss_data_ce):
                    raise ValueError('loss is nan while validating')
                val_ce_loss += loss_data_ce
                val_de_loss += loss_data_de
                predictions = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
                predictions, label = predictions.cpu().detach().numpy(), label.cpu().detach().numpy()
                predictions, label = predictions.astype(np.int32), label.astype(np.int32)
                _ = seg.addBatch(predictions, label)

            val_ce_loss /= len(self.val_loader)
            val_de_loss /= len(self.val_loader)
            pa = seg.classPixelAccuracy()
            IoU = seg.IntersectionOverUnion()
            mIoU = seg.meanIntersectionOverUnion()
            recall = seg.recall()
            mean_IoU = mIoU
            f1_score = (2 * pa * recall) / (pa + recall)
            mean_f1_score = np.mean(f1_score)
            mean_precision = np.mean(pa)
            mean_recall = np.mean(recall)
            print('-' * 10, 'val', '-' * 10)
            print(f'val_Pre: {pa[1]:.4f} '
                  f'val_Recall: {recall[1]:.4f} '
                  f'val_F1_score: {f1_score[1]:.4f} '
                  f'val_IoU: {IoU[1]:.4f}')

            self.writer.add_scalar('val_data/loss_CE', val_ce_loss, self.epoch)
            self.writer.add_scalar('val_data/deloss_CE', val_de_loss, self.epoch)
            self.writer.add_scalar('val_data/val_mIoU', mIoU, self.epoch)
            self.writer.add_scalar('val_data/val_mPrecision', mean_precision, self.epoch)
            self.writer.add_scalar('val_data/val_mRecall', mean_recall, self.epoch)
            self.writer.add_scalar('val_data/val_mF1-score', mean_f1_score, self.epoch)
            for i in range(len(self.classes)):
                self.writer.add_scalar(('val_Precision/' + self.classes[i]), pa[i], self.epoch)
                self.writer.add_scalar(('val_Recall/' + self.classes[i]), recall[i], self.epoch)
                self.writer.add_scalar(('val_IoU/' + self.classes[i]), IoU[i], self.epoch)
                self.writer.add_scalar(('val_F1_score/' + self.classes[i]), f1_score[i], self.epoch)
            metrics.append((val_ce_loss, mIoU, mean_f1_score))
            is_best = mean_IoU > self.best_mean_IoU
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_IoU = mean_IoU

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.s_model.__class__.__name__,
                    'optim_state_dict': self.optim_model.state_dict(),
                    'model_state_dict': self.s_model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_model),
                    'best_mean_IoU': self.best_mean_IoU,
                }, osp.join(self.out, 'best.pth.tar'))
            else:
                if (self.epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.s_model.__class__.__name__,
                        'optim_state_dict': self.optim_model.state_dict(),
                        'model_state_dict': self.s_model.state_dict(),
                        'learning_rate_gen': get_lr(self.optim_model),
                        'best_mean_IoU': self.best_mean_IoU,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                        datetime.now(pytz.timezone(self.time_zone)) -
                        self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] * 5 + \
                      list(metrics) + [elapsed_time] + ['best model epoch: %d' % self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.val_loader)))

            if training:
                self.s_model.train()
            torch.cuda.empty_cache()  # 释放显存

    def train_epoch(self):
        if self.together:
            self.t_model.train()
        self.s_model.train()

        self.running_ce_loss = 0.0
        self.running_de_loss = 0.0
        self.running_kd_loss = 0.0
        self.running_consis_loss = 0.0
        self.running_cross_loss = 0.0
        start_time = timeit.default_timer()
        for batch_idx, sampleS in tqdm.tqdm(
                enumerate(self.loader), total=len(self.loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            metrics = []

            iteration = batch_idx + self.epoch * len(self.loader)
            self.iteration = iteration

            self.optim_model.zero_grad()

            # 1. train  with random images
            for param in self.s_model.parameters():
                param.requires_grad = True

            images = sampleS['img'].cuda()
            label = sampleS['label'].cuda().long()

            # noise = torch.clamp(torch.randn_like(images) * 0.1, -0.5, 0.5)
            # ema_images = images + noise
            if self.together:
                with torch.no_grad():
                    # output_t, _, decoder_output_t = self.t_model(ema_images)
                    # outputT,t_e2,t_d3,_,t_e5 = self.t_model(images)
                    outputT,e4_t = self.t_model(images)

            # t_output = torch.softmax
            # outputS,s_e2,s_d3,outputS_dis,s_e5 = self.s_model(images)
            outputS,e4_s = self.s_model(images)
            # print(outputS.shape)
            outputS_soft = F.softmax(outputS,dim=1)

            # generate pseudo label
            if self.together:
                with torch.no_grad():
                    results = []
                    results_e4 = []
                    for i in range(self.mc_samples):
                        # noise = torch.clamp(torch.randn_like(images) * 0.1, -0.2, 0.2)
                        # t_inputs = images + noise
                        # outputT, _, decoder_outputT = self.t_model(images)
                        t_output,_= self.t_model(images)
                        outputT_soft = torch.softmax(t_output, dim=1)
                        results.append(outputT_soft)

                        # results_e4.append(torch.softmax(t_e4,dim=1))
                        # decoder_outputsT.append(decoder_outputT)

                    results = torch.stack(results, dim=0)  # mc_samples,batch_size,num_classes,h,w
                    entropy = -torch.sum(results * torch.log(results + 1e-12), dim=(0, 2),keepdim=True)  # batch_size,h,w

                    # results_e4 = torch.stack(results_e4, dim=0)
                    # entropy_e4 = - torch.sum(results_e4 * torch.log(results_e4 + 1e-12), dim = (0,2) )
                    # problem: how to use the entropy to modify the error label?
                    # solution 1---linear combination: (-u+1)*p+u*(1-p)
                    # entropy = (entropy - torch.min(entropy)) / (
                    #         torch.max(entropy) - torch.min(entropy))  # no change the data distribution
                    # problem: should we modify the label of the prediction?
                    # pesudo = (1 - entropy) * label + entropy * (1 - label)
                    # pesudo = pesudo.unsqueeze(1).cuda()

                # kd loss
                threshold = (0.75  + 0.25 * sigmoid_rampup(self.iteration, self.max_epoch * len(self.loader))) * np.log(2)
                mask = (entropy < threshold).float()
                consistency_weight = get_current_consistency_weight(self, self.epoch)

                # mask_e4 = (entropy_e4 < threshold).float()

                #学生模型输出和教师模型输出之间作损失
                consistency_dist = 1 * self.consistency_criterion(outputS, outputT) + 3 * self.consistency_criterion(e4_s,e4_t)
                # print(consistency_dist.shape,mask.shape)
                # consistency_dist = 1 * kl_loss(F.log_softmax(outputS,dim=1),F.log_softmax(outputT,dim=1)) + 1 * kl_loss(F.log_softmax(e4_s,dim=1),F.log_softmax(e4_t,dim=1))
                consistency_dist = 0.5 * torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
                consistency_loss = consistency_weight * consistency_dist

                #dis loss
                # dis_to_mask = torch.sigmoid(-1500 * outputS_dis)
                # cross_task_dist = torch.mean((dis_to_mask - F.softmax(outputS,dim=1)) ** 2)
                # cross_task_loss = cross_task_dist * consistency_weight

            # problem: what loss should we use to train the student model?
            loss_ce = ce_loss(outputS, label)
            # print(outputS.shape,label.shape)
            if self.together:
                # loss_de = dice_loss(torch.softmax(outputS, dim=1), pesudo)
                loss_de = dice_loss(outputS_soft[:, 1, :, :], label == 1)
            else:
                loss_de = dice_loss(torch.softmax(outputS, dim=1),label.unsqueeze(1))

            # loss = loss_ce + loss_de * self.lambda_soft

            prediction = torch.argmax(torch.softmax(outputS, dim=1), dim=1).float().cpu()

            self.running_ce_loss += loss_ce.item()
            self.running_de_loss += loss_de.item()

            if self.together:
                self.running_consis_loss += consistency_loss.item()
                # self.running_kd_loss += loss_kd.item()
                # self.running_cross_loss += cross_task_loss.item()
                self.running_kd_loss = 0

            loss_seg_data = loss_ce.data.item()

            if self.together:
            # print(loss_ce,consistency_loss,loss_de,consistency_kd1,consistency_kd2,consistency_weight)
                loss = loss_ce + loss_de * self.lambda_soft + consistency_loss * 5 #+ loss_kd * self.lambda_kd * 1e1 * 5
            else:
                loss = loss_ce + loss_de * self.lambda_soft

            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim_model.step()
            # if self.together:
            #     update_ema_variables(self.s_model, self.t_model, self.ema_decay, self.iteration)
            # self.iter_num = self.iter_num + 1

            # write image log
            if iteration % 10 == 0:  # interval 50 iter writer logs
                images = images[0, ...].clone().cpu().data.numpy()
                self.writer.add_image('image', images, iteration)
                label = label[0, ...].clone().cpu().data
                label = out_to_rgb(label, self.palette, self.classes)
                self.writer.add_image('target', label, iteration)
                prediction = prediction[0, ...].clone().cpu().data
                prediction = out_to_rgb(prediction, self.palette, self.classes)
                self.writer.add_image('prediction', prediction, iteration)

            # if self.epoch > self.warmup_epoch:
            # write train different network or freezn backbone parameter
            self.writer.add_scalar('train/loss_seg', loss_seg_data, iteration)
            metrics.append(loss_seg_data)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                        datetime.now(pytz.timezone(self.time_zone)) -
                        self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + \
                      metrics + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

        self.running_ce_loss /= len(self.loader)
        self.running_de_loss /= len(self.loader)
        self.running_consis_loss /= len(self.loader)
        self.running_kd_loss /= len(self.loader)

        stop_time = timeit.default_timer()

        if self.together:
            loss_all = self.running_ce_loss + self.running_de_loss + self.running_consis_loss
            self.writer.add_scalar('train/loss_all', loss_all, self.epoch)

            print('\n[Epoch: %d] lr:%f,  Average segLoss: %f,  Soft DiceLoss: %f, consis loss: %f, kd loss: %f'
                  'Execution time: %.5f' %
                  (self.epoch, get_lr(self.optim_model), self.running_ce_loss, self.running_de_loss, self.running_consis_loss,self.running_kd_loss,
                   stop_time - start_time))
        else:
            print('\n[Epoch: %d] lr:%f,  Average segLoss: %f,  Soft DiceLoss: %f,'
                  'Execution time: %.5f' %
                  (self.epoch, get_lr(self.optim_model), self.running_ce_loss, self.running_de_loss,
                   stop_time - start_time))



    def train(self):
        for epoch in tqdm.trange(self.epoch + 1, self.max_epoch + 1,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.writer.add_scalar('lr_gen', get_lr(self.optim_model), self.epoch)
            self.train_epoch()
            self.lr_scheduler.step()
            if (self.epoch + 1) % self.interval_validate == 0:
                self.validate()  # 多少轮次验证一次
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break
        self.writer.close()





