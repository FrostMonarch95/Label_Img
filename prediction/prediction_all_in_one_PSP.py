import argparse
import logging
import math
import os
import sys
import zipfile
from collections import OrderedDict
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy import ndimage
from torchvision import transforms

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
__all__ = ['ResNet', 'resnet152', 'BasicBlock', 'Bottleneck']
model_urls = {'resnet152': 'https://hangzh.s3.amazonaws.com/encoding/models/resnet152-0d43d698.zip'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.previous_dialation = previous_dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dilated=True, multi_grid=False,
                 deep_base=True, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=True)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

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


def summary(model, input_shape, batch_size=-1, intputshow=True):
    """
    :param model:
    :param input_shape:
    :param batch_size:
    :param intputshow:
    :return: model infor
    """

    def register_hook(module):
        # noinspection PyArgumentList,PyUnresolvedReferences
        def hook(module_in, inputs):
            class_name = str(module_in.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary_report)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary_report[m_key] = OrderedDict()
            summary_report[m_key]["input_shape"] = list(inputs[0].size())
            summary_report[m_key]["input_shape"][0] = batch_size

            params = 0
            if hasattr(module_in, "weight") and hasattr(module_in.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module_in.weight.size)))
                summary_report[m_key]["trainable"] = module_in.weight.requires_grad
            if hasattr(module_in, "bias") and hasattr(module_in.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module_in.bias.size)))
            summary_report[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)
            and not (module == model)) and 'torch' in str(module.__class__):
            if intputshow is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary_report = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(torch.zeros(input_shape))

    # remove these hooks
    for h in hooks:
        h.remove()

    model_info = ''
    model_info += "-----------------------------------------------------------------------\n"
    line_new = "{:>25}  {:>25} {:>15}".format("Layer (type)", "Input Shape", "Param #")
    model_info += line_new + '\n'
    model_info += "=======================================================================\n"

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary_report:
        line_new = "{:>25}  {:>25} {:>15}".format(
            layer,
            str(summary_report[layer]["input_shape"]),
            "{0:,}".format(summary_report[layer]["nb_params"]),
        )
        total_params += summary_report[layer]["nb_params"]
        if intputshow is True:
            total_output += np.prod(summary_report[layer]["input_shape"])
        else:
            total_output += np.prod(summary_report[layer]["output_shape"])
        if "trainable" in summary_report[layer]:
            if summary_report[layer]["trainable"]:
                trainable_params += summary_report[layer]["nb_params"]

        model_info += line_new + '\n'

    model_info += "=======================================================================\n"
    model_info += "Total params: {0:,}\n".format(total_params)
    model_info += "Trainable params: {0:,}\n".format(trainable_params)
    model_info += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    model_info += "-----------------------------------------------------------------------\n"
    return model_info


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, func):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        func(m)
    if len(c) > 0:
        for layer in c:
            apply_leaf(layer, func)


def set_trainable(layer, b):
    apply_leaf(layer, lambda m: set_trainable_attr(m, b))


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    @staticmethod
    def _make_stages(in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1].split('.')[0]
    cached_file = os.path.join(model_dir, filename + '.pth')
    if not os.path.exists(cached_file):
        cached_file = os.path.join(model_dir, filename + '.zip')
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
        zip_ref = zipfile.ZipFile(cached_file, 'r')
        zip_ref.extractall(model_dir)
        zip_ref.close()
        os.remove(cached_file)
        cached_file = os.path.join(model_dir, filename + '.pth')
    return torch.load(cached_file, map_location=map_location)


def resnet152(pretrained=False, root='./pretrained', **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet152'], model_dir=root))

    return model


class PSPNet(BaseModel):
    # noinspection PyTypeChecker
    def __init__(self, num_classes=8, in_channels=3, use_aux=True,
                 freeze_bn=False, freeze_backbone=False):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        model = resnet152(pretrained=False, norm_layer=norm_layer, )
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux

        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz // 2, m_out_sz // 4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


# noinspection PyAttributeOutsideInit,PyIncorrectDocstring
def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class Prediction:
    def __init__(self, configs):
        self.scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.45734706, 0.43338275, 0.40058118], [0.23965294, 0.23532275, 0.2398498])
        self.num_classes = 8
        self.configs = configs
        self.save_path = configs['training']['save_path']
        self.img_path = configs['training']['img_path']
        self.palette = [0, 0, 0,
                        150, 250, 0,
                        0, 250, 0,
                        0, 100, 0,
                        200, 0, 0,
                        255, 255, 255,
                        0, 0, 200,
                        0, 150, 250]

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # Model
        self.model = PSPNet()

        checkpoint = torch.load(configs['training']['model_path'], map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'module' in list(checkpoint.keys())[0] and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # data
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def predict_pics(self):
        # testing
        with torch.no_grad():
            image1 = Image.open(self.img_path).convert('RGB')
            prediction1 = self.predict(image1)
            # print('set(prediction1)', np.unique(prediction1))
            self.save_images(prediction1, self.save_path, self.img_path)

    def predict(self, image):
        inputs = self.normalize(self.to_tensor(image)).unsqueeze(0)
        prediction = self.multi_scale_predict(inputs)
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

        return prediction

    def multi_scale_predict(self, image, flip=False):
        input_size = (image.size(2), image.size(3))
        upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        total_predictions = np.zeros((self.num_classes, image.size(2), image.size(3)))

        image = image.data.data.cpu().numpy()
        for scale in self.scales:
            scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
            scaled_img = torch.from_numpy(scaled_img).to(self.device)
            scaled_prediction = upsample(self.model(scaled_img).cpu())

            if flip:
                fliped_img = scaled_img.flip(-1).to(self.device)
                fliped_predictions = upsample(self.model(fliped_img).cpu())
                scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
            total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

        total_predictions /= len(self.scales)
        return total_predictions

    def save_images(self, mask, output_path, image_file):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        image_file = os.path.basename(image_file)
        colorized_mask = colorize_mask(mask, self.palette)
        colorized_mask.save(os.path.join(output_path, image_file[:-4] + '_color_label.png'))
        mask = Image.fromarray(np.uint8(mask))
        mask.save(os.path.join(output_path, image_file[:-4] + '_label.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='./config.yml', type=str,
                        help='the config file where you put all the configs')
    args = parser.parse_args()

    with open(args.configs) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    predictor = Prediction(cfg)

    predictor.predict_pics()

    """
    model_path:存放的模型参数的位置
    img_path:需要预测的图片的路径
    save_path:输出图像路径
    """
