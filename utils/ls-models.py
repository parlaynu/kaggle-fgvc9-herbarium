#!/usr/bin/env python3
from itertools import islice
import herbarium.model as m


models = [
    m.mobilenet_v3_small,
    m.mobilenet_v3_large,
    m.resnet50,
    m.resnet101,
    m.resnet152,
    m.efficientnet_v2_s,
    m.efficientnet_v2_m,
    m.efficientnet_v2_l,
    m.convnext_tiny, 
    m.convnext_small, 
    m.convnext_base, 
    m.convnext_large
]


def model_size(model):
    total_size = 0
    for n, p in model.named_parameters():
        p = p.flatten()
        total_size += p.size()[0]
    return total_size


def model_params(model):
    print(model.fullname)
    for n, p in model.named_parameters():
        p = p.flatten()
        print(f"- {n} {p.size()}")
    

def model_modules(model):
    print(model.fullname)
    total_size = 0
    for n, m in model.named_modules():
        print(f"- {n} {type(m)}")


def main():
    for mo in models:
        model = mo(1000, pretrained=False, use_gpu=False)
        size = model_size(model)
        print(f"{model.fullname}: {size:,d}")


if __name__ == "__main__":
    main()

