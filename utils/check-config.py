#!/usr/bin/env python3

def parse_cmdline():
    import sys, os, argparse
    import time
    from datetime import datetime
    import torch
    from herbarium.config import load_config, save_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='configuration file to load', type=str)
    parser.add_argument('variables', help='key=value variables for template expansion', type=str, nargs='*', default=None)
    
    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available()
    
    now = datetime.now()
    
    run_id = now.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("snapshots", f"{run_id}-predict")
    options = {
        'timestamp': now.isoformat(),
        'run_id': run_id,
        'run_dir': run_dir,
        'batch_size': 32,
        'num_workers': 0,
        'num_epochs': 13,
        'use_gpu': False
    }
    if args.variables is not None:
        for kv in args.variables:
            k, v = kv.split('=', maxsplit=1)
            options[k] = v
    
    cfg = load_config(args.config_file, **options)
        
    return cfg, now


def run():
    import sys, os, time
    from datetime import datetime
    import torch
    from herbarium.config import instantiate, build_pipeline
    from herbarium.nodes import iter_fwd, get_root
    from herbarium.utils import progress

    cfg, start = parse_cmdline()

    instances = {}
    for k, v in cfg.items():
        if k.endswith("_pipeline"):
            continue
        instances[k] = instantiate(cfg[k], instances)
    
    if instances.get('model', None) is None:
        raise ValueError("missing model in configuration")
    
    model = instances.get('model')
    print(f"model: {model.fullname}")
    
    data = torch.rand(12, 3, 64, 64)
    output = model(data)
    print(output.shape)
    
    run_dir = cfg['runtime']['run_dir']
    
    for k, _ in cfg.items():
        if not k.endswith("_pipeline"):
            continue
        
        print(f"building {k}")
        pipe = build_pipeline(cfg[k], instances)
        for n in iter_fwd(pipe):
            print(f"- {n.fullname}")
    

if __name__ == "__main__":
    run()

