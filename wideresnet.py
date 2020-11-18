import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class WideBasic(nn.Module):
    def __init__(self, inplanes, outplanes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=0.9)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1, bias=True)
        self.using_dropout = dropout_rate is not None
        if self.using_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(outplanes, momentum=0.9)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.is_identity = stride == 1 and inplanes == outplanes
        if not self.is_identity:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):   # bn-af-conv: PreAct
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.using_dropout:
            out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += x if self.is_identity else self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6

        ch = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        self.conv1 = conv3x3(1, ch[0])
        self.layer1 = self._wide_layer(WideBasic, ch[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, ch[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, ch[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(ch[3], momentum=0.9)
        self.linear = nn.Linear(ch[3], num_classes, bias=True)

        # self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out), inplace=True)
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    from torchsummary import summary
    net = WideResNet(depth=4, widen_factor=2, num_classes=10, dropout_rate=0.2)
    img_shape = (1, 28, 28)
    summary(net, img_shape)
