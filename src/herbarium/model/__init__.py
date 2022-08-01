from .utils import save_state, load_state, load_state_file

from .ensemble import ensemble

from .mobilenet_v3 import mobilenet_v3_small, mobilenet_v3_large
from .efficientnet_v2 import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def load_model(target, num_categories, *, pretrained=True, weights_file=None, use_gpu=True):
    import importlib

    tgt_class_path = target.split('.')
    tgt_class_name = tgt_class_path[-1]
    tgt_module_path = '.'.join(tgt_class_path[0:-1])
    
    tgt_module = importlib.import_module(tgt_module_path)
    tgt_class = getattr(tgt_module, tgt_class_name)

    return tgt_class(num_categories, pretrained=pretrained, weights_file=weights_file, use_gpu=use_gpu)

