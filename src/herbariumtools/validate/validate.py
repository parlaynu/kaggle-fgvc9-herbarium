#!/usr/bin/env python3

def parse_cmdline():
    import os, argparse
    import time
    from datetime import datetime
    import torch
    from herbarium.config import load_config, save_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--use-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=12)
    parser.add_argument('-l', '--batch-limit', help='max batches per epoch (0 = no limit)', type=int, default=0)
    parser.add_argument('-w', '--num-workers', help='number of workers to use', type=int, default=-1)
    parser.add_argument('config_file', help='configuration file to load (- for stdin)', type=str, default=None)
    parser.add_argument('weights_file', help='weights file to load', type=str, default=None)    
    parser.add_argument('variables', help='key=value variables for template expansion', type=str, nargs='*', default=None)
    
    args = parser.parse_args()
    
    use_gpu = not args.use_cpu
    
    if args.num_workers == -1:
        args.num_workers = 4 if use_gpu and torch.cuda.is_available() else 0

    now = datetime.now()
    
    run_id = os.path.basename(args.weights_file)
    run_dir = os.path.dirname(args.weights_file)
    options = {
        'timestamp': now.isoformat(sep=' ', timespec='seconds'),
        'run_id': run_id,
        'run_dir': run_dir,
        'num_epochs': -1,
        'time_limit': -1,
        'batch_limit': args.batch_limit,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'use_gpu': use_gpu,
        'weights_file': args.weights_file
    }
    if args.variables is not None:
        for kv in args.variables:
            k, v = kv.split('=', maxsplit=1)
            options[k] = v
    
    cfg = load_config(args.config_file, **options)
    
    return cfg, now


def run():
    import sys, time
    from itertools import count
    from datetime import datetime
    import torch
    import herbarium
    from herbarium.config import instantiate, build_pipeline
    from herbarium.nodes import iter_fwd, get_root
    from herbarium.utils import progress
    
    cfg, start = parse_cmdline()
    print(f"start at: {start.isoformat(sep=' ', timespec='seconds')}")
    
    instances = {}
    for k, v in cfg.items():
        if k.endswith("_pipeline"):
            continue
        instances[k] = instantiate(cfg[k], instances)
    
    model = instances.get('model', None)
    if model is None:
        raise ValueError("missing model in configuration")
    
    print(f"model: {model.fullname}")

    vpipe = build_pipeline(cfg['validate_pipeline'], instances)
    
    print("validate pipeline:")
    for n in iter_fwd(vpipe):
        if isinstance(n, herbarium.nodes.logger.Logger):
            vpipe = n
            break
        print(f"- {n.fullname}")
    
    print(f"running on {model.device}")

    with torch.no_grad():
        for idx, item in progress(vpipe, header="Vdate", end=""):
            pass
    if metrics := item.get('metrics', None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                continue
            if isinstance(v, float):
                print(f" {k}={v:0.4f}", end="")
            else:
                print(f" {k}={v}", end="")
    print("")
        
