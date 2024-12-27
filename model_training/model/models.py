from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from .cbam import *

# class conv_block_dialted(nn.Module):
#     """
#     Convolution Block
#     """
#
#     def __init__(self, in_ch, out_ch):
#         super(conv_block, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True))
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x


class SegNet(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """
    def __init__(self, num_classes, n_init_features=1, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512)):
        super(SegNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return self.classifier(feat)


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)

# class SegNet(nn.Module):
#     def __init__(self,in_channel=3,out_channel=2):
#         super(SegNet, self).__init__()
#
#         batchNorm_momentum = 0.1
#
#         self.conv11 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
#         self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
#         self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
#
#         self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
#         self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
#
#         self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#
#         self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#
#         self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#
#         self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#
#         self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#
#         self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
#         self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
#
#         self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
#         self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
#
#         self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
#         self.conv11d = nn.Conv2d(64, out_channel, kernel_size=3, padding=1)
#
#
#     def forward(self, x):
#
#         # Stage 1
#         x11 = F.relu(self.bn11(self.conv11(x)))
#         x12 = F.relu(self.bn12(self.conv12(x11)))
#         x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)
#
#         # Stage 2
#         x21 = F.relu(self.bn21(self.conv21(x1p)))
#         x22 = F.relu(self.bn22(self.conv22(x21)))
#         x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)
#
#         # Stage 3
#         x31 = F.relu(self.bn31(self.conv31(x2p)))
#         x32 = F.relu(self.bn32(self.conv32(x31)))
#         x33 = F.relu(self.bn33(self.conv33(x32)))
#         x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)
#
#         # Stage 4
#         x41 = F.relu(self.bn41(self.conv41(x3p)))
#         x42 = F.relu(self.bn42(self.conv42(x41)))
#         x43 = F.relu(self.bn43(self.conv43(x42)))
#         x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)
#
#         # Stage 5
#         x51 = F.relu(self.bn51(self.conv51(x4p)))
#         x52 = F.relu(self.bn52(self.conv52(x51)))
#         x53 = F.relu(self.bn53(self.conv53(x52)))
#         x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)
#
#
#         # Stage 5d
#         x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
#         x53d = F.relu(self.bn53d(self.conv53d(x5d)))
#         x52d = F.relu(self.bn52d(self.conv52d(x53d)))
#         x51d = F.relu(self.bn51d(self.conv51d(x52d)))
#
#         # Stage 4d
#         x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
#         x43d = F.relu(self.bn43d(self.conv43d(x4d)))
#         x42d = F.relu(self.bn42d(self.conv42d(x43d)))
#         x41d = F.relu(self.bn41d(self.conv41d(x42d)))
#
#         # Stage 3d
#         x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
#         x33d = F.relu(self.bn33d(self.conv33d(x3d)))
#         x32d = F.relu(self.bn32d(self.conv32d(x33d)))
#         x31d = F.relu(self.bn31d(self.conv31d(x32d)))
#
#         # Stage 2d
#         x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
#         x22d = F.relu(self.bn22d(self.conv22d(x2d)))
#         x21d = F.relu(self.bn21d(self.conv21d(x22d)))
#
#         # Stage 1d
#         x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
#         x12d = F.relu(self.bn12d(self.conv12d(x1d)))
#         x11d = self.conv11d(x12d)
#
#         return x11d



class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp, self).__init__()

        self.Up = up_conv(in_size, out_size)

        self.Up_conv = conv_block(in_size, out_size)

    def forward(self, inputs1, inputs2):
        # p1 = self.Up(inputs1)
        # p2 = torch.cat((inputs2, p1), dim=1)
        # x = self.Up_conv(p2)

        return self.Up_conv(torch.cat((inputs2,self.Up(inputs1)),dim=1))



class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=2, drop_rate = 0.3):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # self.Conv1_spa = spatial_attention(kernel_size=7)
        # self.Conv2_spa = spatial_attention(kernel_size=7)
        # self.Conv3_spa = spatial_attention(kernel_size=7)
        # self.Conv4_spa = spatial_attention(kernel_size=7)
        # self.Conv1_spa = spatial_attention(kernel_size=7)

        # self.Up5 = up_conv(filters[4], filters[3])
        # self.Up_conv5 = conv_block(filters[4], filters[3])
        self.Up5_conv5 = UnetUp(filters[4], filters[3])

        # self.Up4 = up_conv(filters[3], filters[2])
        # self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Up4_conv4 = UnetUp(filters[3], filters[2])

        # self.Up3 = up_conv(filters[2], filters[1])
        # self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Up3_conv3 = UnetUp(filters[2], filters[1])

        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Up2_conv2 = UnetUp(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.drop1 = nn.Dropout(p = drop_rate)
        # self.drop2 = nn.Dropout(p = 0.3)
    # self.active = torch.nn.Sigmoid()

        # self.crfrnn = crfrnn.CrfRnn(num_labels=2, num_iterations=10)

    def forward(self, x):
        e1 = self.Conv1(x)
        # e1 = self.Conv1_spa(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        # e2 = self.Conv2_spa(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        # e3 = self.Conv3_spa(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        # e4 = self.Conv4_spa(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        e5 = self.drop1(e5)

        d5 = self.Up5_conv5(e5,e4)
        # d5 = self.Up5(e5)
        # d5 = torch.cat((e4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4_conv4(d5,e3)
        # d4 = self.Up4(d5)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.Up3_conv3(d4,e2)
        # d3 = self.Up3(d4)
        # d3 = torch.cat((e2, d3), dim=1)
        # d3 = self.Up_conv3(d3)

        d2 = self.Up2_conv2(d3,e1)
        # d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        # d2 = self.drop2(d2)
        out = self.Conv(d2)

        # crf = self.crfrnn(x, out)
        # d1 = self.active(out)

        return out


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, img_ch=3, output_ch=2, t=2, drop_rate=0.3):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)


        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.dropout1 = nn.Dropout(p = drop_rate)
        self.dropout2 = nn.Dropout(p = drop_rate)
    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        e5 = self.dropout1(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d2 = self.dropout2(d2)
        out = self.Conv(d2)

        # out = self.active(out)

        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, y, x):
        y1 = self.W_g(y)
        x1 = self.W_x(x)
        psi = self.relu(y1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

#空洞卷积模块
class aspp_conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, dilation):
        super(aspp_conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation = dilation, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation = dilation, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


#特征融合模块
class feature_fusion(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=16):
        # 继承父类初始化方法
        super(feature_fusion, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features= in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, x, y):
        # 获取输入特征图的shape
        b, c, h, w = x.shape

        a = x + y
        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        # max_pool = self.max_pool(a)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(a)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        # max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        # x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        # x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        # x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        # x_add = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        s = self.sigmoid(x_avgpool)

        # 调整维度 [b,c]==>[b,c,1,1]
        s= s.view([b, c, 1, 1])

        s_x = x * s
        s_y = y * s
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = s_x + s_y

        return outputs

class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=2, drop_rate = 0.3):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # n2 = 2
        # atrous_rates =[n2 * 16, n2 * 8, n2 * 4, n2 * 2, n2 ]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # self.Conv1_2 = cbam(in_channel = filters[0])
        # self.Conv2_2 = cbam(in_channel = filters[1])
        # self.Conv3_2 = cbam(in_channel = filters[2])
        # self.Conv4_2 = cbam(in_channel = filters[3])
        self.Conv5_2 = cbam(in_channel = filters[4])


        #decoder
        self.Up5 = up_conv(filters[4], filters[3])

        # self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Att5 = feature_fusion(in_channel=filters[3] , ratio=2)

        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        # self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Att4 = feature_fusion(in_channel=filters[2] , ratio=2)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        # self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Att3 = feature_fusion(in_channel=filters[1] , ratio=2)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        # self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Att2 = feature_fusion(in_channel=filters[0] , ratio=2)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.dropout1 = nn.Dropout(p = drop_rate)
        self.dropout2 = nn.Dropout(p = drop_rate)
        self.tanh = nn.Tanh()
        # self.active = torch.nn.Sigmoid()

        self.conv_e5 = conv_block(filters[4],1)
        self.conv_d5 = conv_block(filters[3],1)
        self.conv_d4 = conv_block(filters[2],1)
        self.conv_d3 = conv_block(filters[1],1)
        self.conv_d2 = conv_block(filters[0],1)

    def forward(self, x):
        e1 = self.Conv1(x)
        # e1 = self.Conv1_2(e1)
        # print("e1.shape:{}".format(e1.shape))  8,64,256,256

        # e2 = self.Maxpool1(e1)
        e2 = self.Conv2(self.Maxpool1(e1))
        # e2 = self.Conv2_2(e2)
        # print("e2.shape:{}".format(e2.shape))    8,128,128,128

        # e3 = self.Maxpool2(e2)
        e3 = self.Conv3(self.Maxpool2(e2))
        # e3 = self.Conv3_2(e3)
        # print("e3.shape:{}".format(e3.shape))   8,256,64,64

        # e4 = self.Maxpool3(e3)
        e4 = self.Conv4(self.Maxpool3(e3))
        # e4 = self.Conv4_2(e4)
        # print("e4.shape:{}".format(e4.shape))   8,512,32,32
        # e5 = self.Maxpool4(e4)
        e5 = self.Conv5(self.Maxpool4(e4))
        e5 = self.Conv5_2(e5)

        # print(x5.shape)
        e5 = self.dropout1(e5)
        # print("e5.shape:{}".format(e5.shape))   8,1024,16,16     18
        # d5 = self.Up5(e5)
        # print(d5.shape)
        # x4 = self.Att5(g=d5, x=e4)

        # d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(torch.cat((self.Att5(x=e4,y=self.Up5(e5)), self.Up5(e5)), dim=1))
        # print("d5.shape:{}".format(d5.shape))  8,512,32,32    19
        # d4 = self.Up4(d5)
        # x3 = self.Att4(g=d4, x=e3)
        # d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(torch.cat((self.Att4(x=e3,y =self.Up4(d5)), self.Up4(d5)), dim=1))
        # print("d4.shape:{}".format(d4.shape))   8,256,64,64   18
        # d3 = self.Up3(d4)
        # x2 = self.Att3(g=d3, x=e2)
        # d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(torch.cat((self.Att3(x=e2,y = self.Up3(d4)), self.Up3(d4)), dim=1))
        # print("d3.shape:{}".format(d3.shape))   8,128,128,128  21
        # d2 = self.Up2(d3)
        # x1 = self.Att2(g=d2, x=e1)
        # d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(torch.cat((self.Att2(x=e1, y = self.Up2(d3)), self.Up2(d3)), dim=1))
        # print("d2.shape:{}".format(d2.shape))    8,64,256,256   22
        d2 = self.dropout2(d2)
        out = self.Conv(d2)

        # print("out.shape:{}".format(out.shape))  8,2,256,256

        #  out = self.active(out)
        return out
        # return out,self.conv_d3(e2),self.conv_d3(d3),self.tanh(out),self.conv_e5(e5)


class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, in_ch=3, out_ch=2, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


# For nested 3 channels are required

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


# Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


# Dictioary Unet
# if required for getting the filters and model parameters for each step

class ConvolutionBlock(nn.Module):
    """Convolution block"""

    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, last_active=F.relu):
        super(ConvolutionBlock, self).__init__()

        self.bn = batchnorm
        self.last_active = last_active
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=1)
        self.b1 = nn.BatchNorm2d(out_filters)
        self.c2 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.b2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.c1(x)
        if self.bn:
            x = self.b1(x)
        x = F.relu(x)
        x = self.c2(x)
        if self.bn:
            x = self.b2(x)
        x = self.last_active(x)
        return x


class ContractiveBlock(nn.Module):
    """Deconvuling Block"""

    def __init__(self, in_filters, out_filters, conv_kern=3, pool_kern=2, dropout=0.5, batchnorm=True):
        super(ContractiveBlock, self).__init__()
        self.c1 = ConvolutionBlock(in_filters=in_filters, out_filters=out_filters, kernel_size=conv_kern,
                                   batchnorm=batchnorm)
        self.p1 = nn.MaxPool2d(kernel_size=pool_kern, ceil_mode=True)
        self.d1 = nn.Dropout2d(dropout)

    def forward(self, x):
        c = self.c1(x)
        return c, self.d1(self.p1(c))


class ExpansiveBlock(nn.Module):
    """Upconvole Block"""

    def __init__(self, in_filters1, in_filters2, out_filters, tr_kern=3, conv_kern=3, stride=2, dropout=0.5):
        super(ExpansiveBlock, self).__init__()
        self.t1 = nn.ConvTranspose2d(in_filters1, out_filters, tr_kern, stride=2, padding=1, output_padding=1)
        self.d1 = nn.Dropout(dropout)
        self.c1 = ConvolutionBlock(out_filters + in_filters2, out_filters, conv_kern)

    def forward(self, x, contractive_x):
        x_ups = self.t1(x)
        x_concat = torch.cat([x_ups, contractive_x], 1)
        x_fin = self.c1(self.d1(x_concat))
        return x_fin


class Unet_dict(nn.Module):
    """Unet which operates with filters dictionary values"""

    def __init__(self, n_labels, n_filters=32, p_dropout=0.5, batchnorm=True):
        super(Unet_dict, self).__init__()
        filters_dict = {}
        filt_pair = [3, n_filters]

        for i in range(4):
            self.add_module('contractive_' + str(i), ContractiveBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm))
            filters_dict['contractive_' + str(i)] = (filt_pair[0], filt_pair[1])
            filt_pair[0] = filt_pair[1]
            filt_pair[1] = filt_pair[1] * 2

        self.bottleneck = ConvolutionBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm)
        filters_dict['bottleneck'] = (filt_pair[0], filt_pair[1])

        for i in reversed(range(4)):
            self.add_module('expansive_' + str(i),
                            ExpansiveBlock(filt_pair[1], filters_dict['contractive_' + str(i)][1], filt_pair[0]))
            filters_dict['expansive_' + str(i)] = (filt_pair[1], filt_pair[0])
            filt_pair[1] = filt_pair[0]
            filt_pair[0] = filt_pair[0] // 2

        self.output = nn.Conv2d(filt_pair[1], n_labels, kernel_size=1)
        filters_dict['output'] = (filt_pair[1], n_labels)
        self.filters_dict = filters_dict

    # final_forward
    def forward(self, x):
        c00, c0 = self.contractive_0(x)
        c11, c1 = self.contractive_1(c0)
        c22, c2 = self.contractive_2(c1)
        c33, c3 = self.contractive_3(c2)
        bottle = self.bottleneck(c3)
        u3 = F.relu(self.expansive_3(bottle, c33))
        u2 = F.relu(self.expansive_2(u3, c22))
        u1 = F.relu(self.expansive_1(u2, c11))
        u0 = F.relu(self.expansive_0(u1, c00))
        return F.softmax(self.output(u0), dim=1)

# Need to check why this Unet is not workin properly
#
# class Convolution2(nn.Module):
#     """Convolution Block using 2 Conv2D
#     Args:
#         in_channels = Input Channels
#         out_channels = Output Channels
#         kernal_size = 3
#         activation = Relu
#         batchnorm = True
#
#     Output:
#         Sequential Relu output """
#
#     def __init__(self, in_channels, out_channels, kernal_size=3, activation='Relu', batchnorm=True):
#         super(Convolution2, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernal_size = kernal_size
#         self.batchnorm1 = batchnorm
#
#         self.batchnorm2 = batchnorm
#         self.activation = activation
#
#         self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernal_size,  padding=1, bias=True)
#         self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernal_size, padding=1, bias=True)
#
#         self.b1 = nn.BatchNorm2d(out_channels)
#         self.b2 = nn.BatchNorm2d(out_channels)
#
#         if self.activation == 'LRelu':
#             self.a1 = nn.LeakyReLU(inplace=True)
#         if self.activation == 'Relu':
#             self.a1 = nn.ReLU(inplace=True)
#
#         if self.activation == 'LRelu':
#             self.a2 = nn.LeakyReLU(inplace=True)
#         if self.activation == 'Relu':
#             self.a2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#
#         if self.batchnorm1:
#             x1 = self.b1(x1)
#
#         x1 = self.a1(x1)
#
#         x1 = self.conv2(x1)
#
#         if self.batchnorm2:
#             x1 = self.b1(x1)
#
#         x = self.a2(x1)
#
#         return x
#
#
# class UNet(nn.Module):
#     """Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
#         https://arxiv.org/abs/1505.04597
#         Args:
#             n_class = no. of classes"""
#
#     def __init__(self, n_class, dropout=0.4):
#         super(UNet, self).__init__()
#
#         in_ch = 3
#         n1 = 64
#         n2 = n1*2
#         n3 = n2*2
#         n4 = n3*2
#         n5 = n4*2
#
#         self.dconv_down1 = Convolution2(in_ch, n1)
#         self.dconv_down2 = Convolution2(n1, n2)
#         self.dconv_down3 = Convolution2(n2, n3)
#         self.dconv_down4 = Convolution2(n3, n4)
#         self.dconv_down5 = Convolution2(n4, n5)
#
#         self.maxpool1 = nn.MaxPool2d(2)
#         self.maxpool2 = nn.MaxPool2d(2)
#         self.maxpool3 = nn.MaxPool2d(2)
#         self.maxpool4 = nn.MaxPool2d(2)
#
#         self.upsample1 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample2 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample3 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#         self.dropout4 = nn.Dropout(dropout)
#         self.dropout5 = nn.Dropout(dropout)
#         self.dropout6 = nn.Dropout(dropout)
#         self.dropout7 = nn.Dropout(dropout)
#         self.dropout8 = nn.Dropout(dropout)
#
#         self.dconv_up4 = Convolution2(n4 + n5, n4)
#         self.dconv_up3 = Convolution2(n3 + n4, n3)
#         self.dconv_up2 = Convolution2(n2 + n3, n2)
#         self.dconv_up1 = Convolution2(n1 + n2, n1)
#
#         self.conv_last = nn.Conv2d(n1, n_class, kernel_size=1, stride=1, padding=0)
#       #  self.active = torch.nn.Sigmoid()
#
#
#
#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool1(conv1)
#        # x = self.dropout1(x)
#
#         conv2 = self.dconv_down2(x)
#         x = self.maxpool2(conv2)
#        # x = self.dropout2(x)
#
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool3(conv3)
#        # x = self.dropout3(x)
#
#         conv4 = self.dconv_down4(x)
#         x = self.maxpool4(conv4)
#         #x = self.dropout4(x)
#
#         x = self.dconv_down5(x)
#
#         x = self.upsample4(x)
#         x = torch.cat((x, conv4), dim=1)
#         #x = self.dropout5(x)
#
#         x = self.dconv_up4(x)
#         x = self.upsample3(x)
#         x = torch.cat((x, conv3), dim=1)
#        # x = self.dropout6(x)
#
#         x = self.dconv_up3(x)
#         x = self.upsample2(x)
#         x = torch.cat((x, conv2), dim=1)
#         #x = self.dropout7(x)
#
#         x = self.dconv_up2(x)
#         x = self.upsample1(x)
#         x = torch.cat((x, conv1), dim=1)
#         #x = self.dropout8(x)
#
#         x = self.dconv_up1(x)
#
#         x = self.conv_last(x)
#      #   out = self.active(x)
#
#         return x