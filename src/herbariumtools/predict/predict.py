#!/usr/bin/env python3

def parse_cmdline():
    import sys, os, argparse
    import time
    from datetime import datetime
    import torch
    from herbarium.config import load_config, save_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--use-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=12)
    parser.add_argument('-l', '--batch-limit', help='max number of images to predict', type=int, default=0)
    parser.add_argument('-w', '--num-workers', help='number of workers to use', type=int, default=-1)
    parser.add_argument('-p', '--pattern', help='image glob specification', type=str, default=None)
    parser.add_argument('config_file', help='configuration file to load', type=str, default=None)
    parser.add_argument('weights_file', help='weights file to load', type=str, default=None)
    parser.add_argument('variables', help='key=value variables for template expansion', type=str, nargs='*', default=None)
    
    args = parser.parse_args()
    
    use_gpu = not args.use_cpu
    if args.num_workers == -1:
        args.num_workers = 4 if use_gpu and torch.cuda.is_available() else 0
    
    now = datetime.now()
    
    run_id = now.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.dirname(args.weights_file)

    options = {
        'timestamp': now.isoformat(),
        'run_id': run_id,
        'run_dir': run_dir,
        'num_workers': args.num_workers,
        'batch_size': args.batch_size,
        'batch_limit': args.batch_limit,
        'weights_file': args.weights_file,
        'use_gpu': use_gpu
    }
    if args.pattern is not None:
        options['pattern'] = args.pattern
    if args.variables is not None:
        for kv in args.variables:
            k, v = kv.split('=', maxsplit=1)
            options[k] = v
    
    cfg = load_config(args.config_file, **options)
    
    run_dir = cfg['runtime']['run_dir']
    os.makedirs(run_dir, mode=0o777, exist_ok=True)
    save_config(cfg, run_dir, "predict")
    
    return cfg, now


def run():
    import sys, os, time
    from datetime import datetime
    import torch
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
    
    if instances.get('model', None) is None:
        raise ValueError("missing model in configuration")
    
    model = instances.get('model')               # required at top level of config
    print(f"model: {model.fullname}")
        
    run_dir = cfg['runtime']['run_dir']
    
    ppipe = build_pipeline(cfg['predict_pipeline'], instances)
    
    print("predict pipeline:")
    for n in iter_fwd(ppipe):
        print(f"- {n.fullname}")

    print(f"running on {model.device}")
    
    with torch.no_grad():
        results = {}
        for idx, item in progress(ppipe, header="Predict"):
            image_ids = item['image_id']
            predictions = item['prediction']
            for image_id, prediction in zip(image_ids, predictions):
                results[int(image_id)] = int(prediction)
    
    pred_file = os.path.splitext(os.path.basename(cfg['runtime']['weights_file']))[0] + "-pred.csv"
    pred_file = os.path.join(run_dir, pred_file)
    print(f"saving to {pred_file}")
    with open(pred_file, "w") as f:
        print("Id,Predicted", file=f)
        keys = list(results.keys())
        keys.sort()
        for key in keys:
            print(f"{key},{results[key]}", file=f)
        
    end = datetime.now()
    print(f"finish at: {end.isoformat(sep=' ', timespec='seconds')}")
    
    duration = end - start
    print(f"run time: {duration}")

