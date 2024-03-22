# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet34, resnet18
from PIL import Image

_encoders = {'resnet34' : resnet34, 'resnet18' : resnet18, }
_transforms = {
	'resnet34' :
		transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'resnet18' :
		transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

class Encoder(nn.Module):
    def __init__(self, encoder_type):
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity() # fc layer is replaced with identity

    def forward(self, x):
        x = self.model(x)
        return x

    # the transform is resize - center crop - normalize (imagenet normalize) No data aug here
    def get_transform(self):
        return _transforms[self.encoder_type]

    def get_features(self, x):
        with torch.no_grad():
            z = self.model(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/

class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()

    def forward(self, x):
        return x

    def get_transform(self):
        return transforms.Compose([
                          transforms.ToTensor(),
                          ])

    def get_features(self, x):
        return x.reshape(-1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
