from collections import OrderedDict

import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, input_channel, num_classes,
                 channels=[8, 16, 16, 32],
                 strides=[1, 2, 1, 2],          # 28x28 =stride1=> 28x28 =stride2=> 14x14 => 14x14 => 7x7
                 dropout_p=None
                 ):
        super(ConvNet, self).__init__()
        self.using_dropout = dropout_p is not None
        self.num_layers = len(self.channels)
        assert len(self.strides) == self.num_layers
        
        self.bn0 = nn.BatchNorm2d(input_channel)
        self.backbone = self._make_backbone(
            in_channels=[input_channel] + channels[:-1],
            out_channels=channels,
            strides=strides
        )
        if self.using_dropout:
            self.dropout_p = dropout_p
            self.dropout = nn.Dropout(p=dropout_p)
        
        last_ch = round(channels[-1] * 1.5)
        self.conv_last = nn.Sequential(
            nn.Conv2d(channels[-1], last_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(last_ch, num_classes, bias=True)
        
    def forward(self, x):
        x = self.bn0(x)                     # x.shape: (B, C, H, W)
        features = self.backbone(x)         # features.shape: (B, channels[-1], 7, 7)
        features = self.conv_last(features) # features.shape: (B, last_ch, 1, 1)
        if self.using_dropout:
            features = self.dropout(features)
        features = features.view(features.size(0), -1)  # features.shape: (B, last_ch)
        logits = self.classifier(features)              # logits.shape: (B, num_classes)
        return logits
    
    def _make_backbone(self, in_channels, out_channels, strides):
        backbone = OrderedDict()
        for i in range(self.num_layers):
            backbone[f'conv2d_{i}'] = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels[i], kernel_size=3,
                          stride=strides[i], padding=1, bias=False),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(backbone)
