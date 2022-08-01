#!/usr/bin/env python3

def parse_cmdline():
    import argparse
    import torch
    from herbarium.config import load_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--use-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=12)
    parser.add_argument('-l', '--batch-limit', help='max batches per epoch (0 = no limit)', type=int, default=0)
    parser.add_argument('-w', '--num-workers', help='number of workers to use', type=int, default=-1)
    parser.add_argument('config_file', help='configuration file to load (- for stdin)', type=str, default=None)
    parser.add_argument('model_class', help='the class of the model being loaded', type=str, default=None)
    parser.add_argument('weights_files', help='the weights files to load and convert to an swa model', type=str, nargs='+', default=None)
    
    args = parser.parse_args()
    
    use_gpu = not args.use_cpu
    if args.num_workers == -1:
        args.num_workers = 4 if use_gpu and torch.cuda.is_available() else 0
    
    options = {
        "use_gpu": use_gpu,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "batch_limit": args.batch_limit,
    }

    cfg = load_config(args.config_file, **options)
    cfg['model_class'] = args.model_class
    cfg['weights_files'] = args.weights_files

    return cfg


def run():
    import sys, os, time
    import torch
    from torch.optim import swa_utils
    from herbarium.config import build_pipeline
    from herbarium.model import load_model, load_state_file, save_state
    from herbarium.utils import progress
    
    cfg = parse_cmdline()

    use_gpu = cfg['runtime']['use_gpu']
    model_class = cfg['model_class']
    weights_files = cfg['weights_files']
    num_categories = cfg['runtime']['num_categories']

    model = load_model(model_class, num_categories, use_gpu=use_gpu)
    device = model.device
    print(f"running on {device}")
    
    swa_model = swa_utils.AveragedModel(model)
    
    for weights_file in weights_files:
        print(f"loading {weights_file}")
        _, model, _ = load_state_file(model, None, weights_file, device=device, verbose=False)
        swa_model.update_parameters(model)

    print("building swa bn pipeline")
    swapipe = build_pipeline(cfg['swa_pipeline'])
    for idx, item in progress(swapipe, header="SWA"):
        inputs = item['image'].to(device, non_blocking=True)

        with torch.no_grad():
            outputs = swa_model(inputs)
    
    state_dir = os.path.dirname(weights_files[-1])
    name = os.path.splitext(os.path.basename(weights_files[-1]))[0] + "-swa"
    fullpath = os.path.join(state_dir, name)

    print(f"saving model to {fullpath}")
    save_state(0, swa_model, None, name, state_dir=state_dir, verbose=False, overwrite=True)

