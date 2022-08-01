import torch
import torch.nn as nn
import torch.optim.swa_utils as swa_utils
import torchvision.models

from .utils import load_state_file


def _tweak_resnet(model, num_categories, weights_file, use_gpu):

    # update the final classifier
    out_features = model.fc.out_features
    if num_categories is not None and num_categories != out_features:
        in_features = model.fc.in_features
        newfc = nn.Linear(in_features, num_categories)
        model.fc = newfc
    
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


def resnet18(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False
    
    weights = None
    if pretrained:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

    model = torchvision.models.resnet18(weights=weights)
    model = _tweak_resnet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.resnet18"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def resnet34(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False

    weights = None
    if pretrained:
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1

    model = torchvision.models.resnet34(weights=weights)
    model = _tweak_resnet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.resnet34"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def resnet50(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False

    weights = None
    if pretrained:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2

    model = torchvision.models.resnet50(weights=weights)
    model = _tweak_resnet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.resnet50"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def resnet101(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False

    weights = None
    if pretrained:
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2

    model = torchvision.models.resnet101(weights=weights)
    model = _tweak_resnet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.resnet101"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model


def resnet152(num_categories, *, pretrained=True, weights_file=None, use_gpu=False):
    pretrained = pretrained if weights_file is None else False

    weights = None
    if pretrained:
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2

    model = torchvision.models.resnet152(weights=weights)
    model = _tweak_resnet(model, num_categories, weights_file, use_gpu)

    model.fullname = "herbarium.model.resnet152"
    if isinstance(model, swa_utils.AveragedModel):
        model.fullname += ".swa"

    return model

