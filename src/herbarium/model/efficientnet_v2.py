import torch
import torch.nn as nn
import torch.optim.swa_utils as swa_utils
import torchvision.models

from .utils import load_state_file


def _tweak_efficientnet_v2(model, num_categories, weights_file, use_gpu):

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
    
    return model


def efficientnet_v2_s(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False
    
    weights = None
    if pretrained:
        weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1

    model = torchvision.models.efficientnet_v2_s(weights=weights)
    model = _tweak_efficientnet_v2(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.efficientnet_v2_s"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def efficientnet_v2_m(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False
    
    weights = None
    if pretrained:
        weights = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1

    model = torchvision.models.efficientnet_v2_m(weights=weights)
    model = _tweak_efficientnet_v2(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.efficientnet_v2_m"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def efficientnet_v2_l(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False
    
    weights = None
    if pretrained:
        weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1

    model = torchvision.models.efficientnet_v2_l(weights=weights)
    model = _tweak_efficientnet_v2(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.efficientnet_v2_l"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model
