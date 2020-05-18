import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as standard_transforms
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import os
import gdal
from tqdm import tqdm
import math

"""====================model======================="""

import torch.nn as nn


class DeepLabV3(nn.Module):
    def __init__(self, model_id, cls_nums):
        super(DeepLabV3, self).__init__()

        self.num_classes = cls_nums

        self.model_id = model_id
        # self.project_dir = project_dir
        # self.create_model_dirs()

        self.resnet = ResNet18_OS8()  # NOTE! specify the type of ResNet here
        self.aspp = ASPP(
            num_classes=self.num_classes)  # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(
            x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.interpolate(output, size=(h, w), mode="bilinear",
                               align_corners=True)  # (shape: (batch_size, num_classes, h, w))

        return output


class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))  # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map)  # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear",
                                align_corners=True)  # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img],
                        1)  # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))  # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out)  # (shape: (batch_size, num_classes, h/16, w/16))

        return out


class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes):
        super(ASPP_Bottleneck, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(4 * 512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(4 * 512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(4 * 512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(4 * 512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(4 * 512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 4*512, h/16, w/16))

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))  # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map)  # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear",
                                align_corners=True)  # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img],
                        1)  # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))  # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out)  # (shape: (batch_size, num_classes, h/16, w/16))

        return out


"""=======resnet18==========="""

import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18']


# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock1, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


"""============resnet============="""


def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)  # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks)  # (*blocks: call with unpacked list entires as arguments)

    return layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(
            x)))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(
            out))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(
            x)  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(
            out)  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x)))  # (shape: (batch_size, channels, h, w))
        out = F.relu(self.bn2(self.conv2(
            out)))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(
            out))  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(
            x)  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.relu(
            out)  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out


class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_BasicBlock_OS8, self).__init__()

        if num_layers == 18:
            # resnet = models.resnet18()
            """自己改过，重新定义了一个resnet18"""
            resnet = resnet18()

            # load pretrained model:
            """自己改过原本要加载预训练模型"""
            # resnet.load_state_dict(torch.load("../pretrained_models/resnet/resnet18-5c106cde.pth"))

            # remove fully connected layer, avg pool, layer4 and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2
            # print("pretrained resnet, 18")
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1,
                                 dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1,
                                 dilation=4)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c3 = self.resnet(x)  # (shape: (batch_size, 128, h/8, w/8)) (it's called c3 since 8 == 2^3)

        output = self.layer4(c3)  # (shape: (batch_size, 256, h/8, w/8))
        output = self.layer5(output)  # (shape: (batch_size, 512, h/8, w/8))

        return output


def ResNet18_OS8():
    return ResNet_BasicBlock_OS8(num_layers=18)


"""=======================着色函数=================="""
palette = [0, 0, 0,
           200, 0, 0,
           0, 0, 200,
           0, 100, 0,
           0, 250, 0,
           150, 250, 0,
           255, 255, 255,
           0, 150, 250]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask  ###Image
def stretch_n(bands, lower_percent=2, higher_percent=98):
        # print(bands.dtype)
        # 一定要使用float32类型，原因有两个：1、Keras不支持float64运算；2、float32运算要好于uint16
        out = np.zeros_like(bands).astype(np.float32)
        # print(out.dtype)
        for i in range(bands.shape[2]):
            # 这里直接拉伸到[0,1]之间，不需要先拉伸到[0,255]后面再转
            a = 0
            b = 1
            # 计算百分位数（从小到大排序之后第 percent% 的数）
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t

        return out

def tiff2numpy(tiff_path):
    if tiff_path.endswith('.tiff') or tiff_path.endswith('.tif'):
        tiff = gdal.Open(tiff_path)
        im_width = tiff.RasterXSize  # 栅格矩阵的列数
        im_height = tiff.RasterYSize  # 栅格矩阵的行数
        # im_bands = tiff.RasterCount  # 波段数
        tiff_data = tiff.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        # im_geotrans = tiff.GetGeoTransform()  # 获取仿射矩阵信息
        # im_blueBand = tiff_data[0, 0:im_height, 0:im_width]  # 获取蓝波段
        # im_greenBand = tiff_data[1, 0:im_height, 0:im_width]  # 获取绿波段
        # im_redBand = tiff_data[2, 0:im_height, 0:im_width]  # 获取红波段
        # im_nirBand = tiff_data[3, 0:im_height, 0:im_width]  # 获取近红外波段
        if np.max(tiff_data)>255:
            tiff_data=np.uint8(stretch_n(np.float32(tiff_data.transpose(1,2,0)))*255)
            tiff_data=tiff_data.transpose(2,0,1)
        return im_width, im_height, tiff_data  ###numpy格式,shape(C*H*W)
    else:
        print("filepath error:please check you input tiff_image")


def data_process(img):
    # mean_std = ([0.485, 0.456, 0.406, 0.450], [0.229, 0.224, 0.225, 0.225])
    mean_std = ([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),  ##把HWC转换为chw
        standard_transforms.Normalize(*mean_std)
    ])
    img_Normalize = input_transform(img)
    img_Normalize1 = img_Normalize.unsqueeze(0)
    return img_Normalize1  ###shape(1*C*H*W)


def prediction_small_image(net, im_data):
    net = net.cuda()
    net.eval()
    img = Variable(im_data).cuda()
    outputs = net(img)
    predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

    predictions_pil = colorize_mask(predictions)
    predictions = Image.fromarray(np.uint8(predictions)).convert('P')
    predictions = np.array(predictions)

    return predictions


def img_padding(img, patch_size):
    """
    :param img:numpy object
    :param padding_size: padding size
    :return: new img after padding
    """
    print(img.shape)  # CHW
    _, h, w = img.shape
    padding_size = ((math.ceil(w / patch_size)) * patch_size - w, (math.ceil(h / patch_size)) * patch_size - h)
    new_image = np.pad(img, ((0, 0), (0, padding_size[1]), (0, padding_size[0])), 'constant', constant_values=0)
    # print(new_image.shape)
    return new_image


def load_model(model_path, mode=2, cls_nums=2):
    ##mode=1或者2,mode为1代表多分类.mode为2代表二分类
    assert mode in [1, 2]
    if mode == 1:
        cls_nums = cls_nums
    else:
        cls_nums = 2  ###多分类类别数
    net = DeepLabV3("1", cls_nums=cls_nums)

    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net


def swell_prediction(model_path, tiff_img_path, save_img_path,patch_size=256, mode=2, cls_nums=2):

    """
    :param model_path:模型权重文件
    :param tiff_img_path: tiff文件路径,8位,4通道tiff
    :param save_img_path: 预测图保存路径
    :param patch_size: 膨胀预测的size
    :param mode: 模式
    :param cls_nums: 分类类别数,多分类为8
    :return:
    """
    print("starting predicting tiff")
    mode_dict = {"1": cls_nums, "2": 2}

    im_width, im_height, im_data = tiff2numpy(tiff_img_path)

    net = load_model(model_path, mode=mode, cls_nums=mode_dict[str(mode)])


    if im_width < 3*patch_size or im_height < 3*patch_size:
        img = data_process(im_data)
        predictions_numpy = prediction_small_image(net, im_data=img)
    else:
        im_data = img_padding(im_data, patch_size=patch_size)
        c, h, w = im_data.shape
        im_data = np.pad(im_data,((0, 0), (patch_size, patch_size), (patch_size, patch_size)), 'constant',
                         constant_values=0)
        x_len = int(w / patch_size)
        y_len = int(h / patch_size)
        predictions_numpy = np.zeros((h, w)).astype(np.uint8)
        for i in tqdm(range(x_len)):
            for j in range(y_len):
                img = im_data[:, j * patch_size:(j+3) * patch_size,
                      i * patch_size:(i + 3) * patch_size].transpose(1, 2, 0)
                img = data_process(img)
                predictions_ = prediction_small_image(net, img)[patch_size:2*patch_size,patch_size:2*patch_size]
                predictions_numpy[j * patch_size:(j + 1) * patch_size,
                i * patch_size:(i + 1) * patch_size] = predictions_
        predictions_numpy = predictions_numpy[0:im_height, 0:im_width]
        predictions_numpy = predictions_numpy[0:im_height, 0:im_width]
    Image.fromarray(predictions_numpy).convert("P").save(save_img_path.replace(".png", "_label.png"))
    predictions_img = colorize_mask(predictions_numpy)
    predictions_img.save(save_img_path)
    print("finish inference image:{}".format(save_img_path.split("/")[-1]))
#input_img_path 输入图片的路径 例如 ：D:/aaa.jpg
#save_img_path 保存的图片的路径最后带有 例如：D:/label/
def deep_learning_to_label(input_img_path,save_img_path):
    print("deep learning ")
    
    model_path="epoch_178_acc_0.80090_acc-cls_0.69342_mean-iu_0.45412.pth"  #默认直接读取相对路径的pth 后面可以修改
    tiff_img_path=input_img_path
    
    save_img_path=save_img_path+input_img_path.split('/')[-1].replace(".tiff","_muti.png").replace(".tif",".png")
    print(tiff_img_path)
    print("here")
    print(save_img_path)
    swell_prediction(model_path=model_path, tiff_img_path=tiff_img_path, save_img_path=save_img_path,patch_size=256, mode=1,
                     cls_nums=8)
               
if __name__ == "__main__":
    model_path = "./pretrained_model/models_muti_cls/epoch_178_acc_0.80090_acc-cls_0.69342_mean-iu_0.45412.pth"
    tiff_img_root = "./tiff_img"
    seg_img_root = "./seg_img"
    tiff_img_name = "GF2_PMS1_E116.2_N39.2_20180131_L1A0002971148-MSS1.tiff"
    # tiff_img_name = "GF2_PMS1_E104.2_N31.0_20171210_L1A0002836824-MSS1_CUT_1.tif"
    # tiff_img_name = "116_projection_CUT.tif"

    tiff_img_path = os.path.join(tiff_img_root, tiff_img_name)
    save_img_path = os.path.join(seg_img_root, tiff_img_name.replace(".tiff", ".png").replace(".tif",".png"))

    swell_prediction(model_path=model_path, tiff_img_path=tiff_img_path, save_img_path=save_img_path,patch_size=256, mode=1,
                     cls_nums=8)

    """
    使用了膨胀预测，对任意尺度的图像都可以预测，但是输入凸显个必须是8位4通道tiff不能是16通道
    model_path:存放的模型参数的位置
    tiff_img_path:需要预测的图片的路径,最好是绝对路径
    save_img_path:输出图像路径
    mode:   可以为1(代表多分类,此时cls_nums必须为正确的值比如8,若要更换类别数,那么需要对应的模型参数才行)
            也可以为2(代表二分类,此时cls_nums锁定为2)
    cls_nums:分类类别数
    注意::多分类的时候把data_process函数中,(整个文件507行)mean_std用mean_std = ([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    """
