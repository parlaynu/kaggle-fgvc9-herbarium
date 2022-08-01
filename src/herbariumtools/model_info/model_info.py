import argparse
from herbarium.model import load_model


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_class', help='the class of the model being loaded', type=str, default=None)
    parser.add_argument('num_categories', help='number of categories', type=int, nargs='?', default=None)
    
    args = parser.parse_args()
    
    return args.model_class, args.num_categories


def run():
    
    mclass, ncats = parse_cmdline()
    
    model = load_model(mclass, ncats, pretrained=False, use_gpu=False)
    print(model)
    
    if hasattr(model, "named_param_groups"):
        for (n, p), p2 in zip(model.named_param_groups(), model.param_groups()):
            p = list(p)
            p2 = list(p2)
            print(f"- {n}: {len(p)} {len(p2)}")
    

