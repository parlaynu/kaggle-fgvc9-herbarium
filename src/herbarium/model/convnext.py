from itertools import islice, chain
from types import MethodType

import torch
import torch.nn as nn
import torch.optim.swa_utils as swa_utils
import torchvision.models

from .utils import load_state_file


def _tweak_mobilenet(model, num_categories, weights_file, use_gpu):

    # update the final classifier
    out_features = model.classifier[-1].out_features
    if num_categories is not None and num_categories != out_features:
        in_features = model.classifier[-1].in_features
        newfc = nn.Linear(in_features, num_categories)
        model.classifier[-1] = newfc
    
    # load any weights file
    if weights_file is not None:
        _, model, _ = load_state_file(model, None, weights_file)
    
    # move the model to the device
    device = torch.device('cuda') if (use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    model = model.to(device)

    # set extra attributes on the model
    model.device = device
    model.num_outputs = num_categories
    
    model.named_param_groups = MethodType(_named_param_groups, model)
    model.param_groups = MethodType(_param_groups, model)
    
    return model


def _named_param_groups(self):
    groups = []

    prefix = None
    params = []

    for n, p in self.named_parameters():
        if n.startswith("classifier"):
            cprefix = "classifier"
        else:
            cprefix = ".".join(islice(n.split("."), 2))
        
        if prefix is not None and cprefix != prefix:
            groups.append((prefix, chain(params)))
            params = []
        
        params.append(p)
        prefix = cprefix
    
    groups.append((prefix, chain(params)))
    
    return reversed(groups)


def _param_groups(self):
    return [p for _, p in self.named_param_groups()]


def convnext_tiny(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False
    
    weights = None
    if pretrained:
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1

    model = torchvision.models.convnext_tiny(weights=weights)
    model = _tweak_mobilenet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.convnext_tiny"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def convnext_small(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False

    weights = None
    if pretrained:
        weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1

    model = torchvision.models.convnext_small(weights=weights)
    model = _tweak_mobilenet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.convnext_small"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def convnext_base(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False

    weights = None
    if pretrained:
        weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1

    model = torchvision.models.convnext_base(weights=weights)
    model = _tweak_mobilenet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.convnext_base"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def convnext_large(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False

    weights = None
    if pretrained:
        weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1

    model = torchvision.models.convnext_large(weights=weights)
    model = _tweak_mobilenet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.convnext_large"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model

