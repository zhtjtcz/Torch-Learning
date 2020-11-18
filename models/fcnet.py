from collections import OrderedDict

import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self, input_dim, output_dim,
                 hid_dims=[72, 36, 18],
                 dropout_p=None
                 ):
        super(FCNet, self).__init__()
        self.using_dropout = dropout_p is not None
        
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.backbone = self._make_backbone(
            len(hid_dims),
            [input_dim] + hid_dims[:-1],
            hid_dims
        )
        if self.using_dropout:
            self.dropout_p = dropout_p
            self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(hid_dims[-1], output_dim, bias=True)
    
    def forward(self, x):                   # x.shape:  (B, C, H, W)
        flatten_x = x.view(x.size(0), -1)   # flatten_x.shape: (B, C*H*W)
        flatten_x = self.bn0(flatten_x)
        features = self.backbone(flatten_x) # features.shape: (B, hid_dims[-1])
        if self.using_dropout:
            features = self.dropout(features)
        logits = self.classifier(features)  # logits.shape: (B, output_dim)
        return logits
    
    def _make_backbone(self, num_layers, in_dims, out_dims):
        backbone = OrderedDict()
        for i in range(num_layers):
            backbone[f'linear_{i}'] = nn.Sequential(
                nn.Linear(in_dims[i], out_dims[i], bias=False),
                nn.BatchNorm1d(out_dims[i]),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(backbone)
