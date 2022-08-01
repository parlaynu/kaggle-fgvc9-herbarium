#!/usr/bin/env python3

def parse_cmdline():
    import os, argparse
    import time
    from datetime import datetime
    import torch
    from herbarium.config import load_config, save_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--use-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-t', '--time-limit', help='max runtime in minutes (0 = no limit)', type=int, default=0)
    parser.add_argument('-e', '--num-epochs', help='number of epochs (0 = no limit)', type=int, default=0)
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=12)
    parser.add_argument('-l', '--batch-limit', help='max batches per epoch (0 = no limit)', type=int, default=0)
    parser.add_argument('-w', '--num-workers', help='number of workers to use', type=int, default=-1)
    parser.add_argument('-r', '--run-id', help='the id of the run', type=str, default=None)
    parser.add_argument('config_file', help='configuration file to load (- for stdin)', type=str, default=None)
    parser.add_argument('variables', help='key=value variables for template expansion', type=str, nargs='*', default=None)
    
    args = parser.parse_args()
    
    use_gpu = not args.use_cpu
    
    if args.num_workers == -1:
        args.num_workers = 4 if use_gpu and torch.cuda.is_available() else 0

    if args.time_limit == 0 and args.num_epochs == 0:
        args.num_epochs = 1
    
    now = datetime.now()
    
    run_id = args.run_id
    if run_id is None:
        run_id = now.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("snapshots", f"{run_id}-train")

    options = {
        'timestamp': now.isoformat(sep=' ', timespec='seconds'),
        'run_id': run_id,
        'run_dir': run_dir,
        'num_epochs': args.num_epochs,
        'time_limit': args.time_limit,
        'batch_limit': args.batch_limit,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'use_gpu': use_gpu
    }
    
    if args.variables is not None:
        for kv in args.variables:
            k, v = kv.split('=', maxsplit=1)
            options[k] = v
    
    cfg = load_config(args.config_file, **options)
    
    run_dir = cfg['runtime']['run_dir']
    os.makedirs(run_dir, mode=0o777, exist_ok=False)
    save_config(cfg, run_dir, "train")
    
    return cfg, now


def check_data(tpipe, vpipe):
    from herbarium.nodes import get_root
    
    ok = True
    
    troot = get_root(tpipe)
    timages = set(troot.image_ids())
    if len(timages) != len(troot):
        print("Error: duplicates in train images")
        ok = False
    
    vroot = get_root(vpipe)
    vimages = set(vroot.image_ids())
    if len(vimages) != len(vroot):
        print("Error: duplicates in validate images")
        ok = False

    if len(timages.intersection(vimages)) != 0:
        overlap = timages.intersection(vimages)
        print(f"Error: train and validate datasets overlap: {len(tnames.intersection(vnames))} {overlap.pop()}")
        ok = False

    if ok == False:
        raise ValueError("unable to proceed: resolve dataset issues")


def snapshot_state(snapshot_dir, epoch, model, optimizer, metrics):
    import os.path
    from herbarium.model import save_state
    
    snapshot_base = os.path.basename(snapshot_dir)
    name = f'{snapshot_base}-{epoch:02d}'
    
    m = model.wrapped if hasattr(model, "wrapped") else model
    save_state(epoch, m, optimizer, name, state_dir=snapshot_dir, verbose=True, overwrite=True)
    
    if metrics is None:
        return
    
    f1_scores = metrics.get("f1_scores", None)
    f1_data = metrics.get("f1_data", None)
    if f1_scores is None or f1_data is None:
        return
    
    f1_data = f1_data.T
    path = os.path.join(snapshot_dir, f"{name}-f1.csv")
    with open(path, "w") as f:
        print("category,f1,tp,fp+fn", file=f)
        for idx, row in enumerate(zip(f1_scores,f1_data)):
            print(f"{idx},{row[0]},{row[1][0]},{row[1][1]}", file=f)


def run():
    import sys, time
    from itertools import count
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
    optimizer = instances.get('optimizer', None) # optional at top level of config
    
    num_epochs = cfg['runtime']['num_epochs']
    max_runtime = cfg['runtime']['time_limit']
    
    run_dir = cfg['runtime']['run_dir']
    
    tpipe = build_pipeline(cfg['train_pipeline'], instances)
    troot = get_root(tpipe)
    vpipe = build_pipeline(cfg['validate_pipeline'], instances)
    
    check_data(tpipe, vpipe)
    
    snapshot_enabled = cfg['runtime']['snapshots']['enabled']
    snapshot_start = cfg['runtime']['snapshots']['start']
    snapshot_rate = cfg['runtime']['snapshots']['rate']
    last_snapshot = -1
    
    save_best = cfg['runtime']['save_best']['enabled']
    save_best_start = cfg['runtime']['save_best']['start']
    save_best_value = sys.maxsize
    
    print(f"model: {model.fullname}")
    
    print("train pipeline:")
    for n in iter_fwd(tpipe):
        print(f"- {n.fullname}")

    print("validate pipeline:")
    for n in iter_fwd(vpipe):
        print(f"- {n.fullname}")

    print(f"snapshots: {'enabled' if snapshot_enabled else 'disabled'}")
    print(f"save best: {'enabled' if save_best else 'disabled'}")
    
    if num_epochs > 0:
        print(f"num epochs: {num_epochs}")
    if max_runtime > 0:
        print(f"max runtime: {max_runtime} minutes")

    use_amp = cfg['runtime'].get('use_amp', False)
    print(f"amp: {'enabled' if use_amp else 'disabled'}")
    print(f"running on {model.device}")

    startstamp = time.time()
    for epoch in count():
        if num_epochs > 0 and epoch >= num_epochs:
            print("epoch limit reached")
            break
        if max_runtime > 0:
            elapsed = (time.time() - startstamp)/60
            if elapsed > max_runtime:
                print("runtime limit reached")
                break

        troot.shuffle()
        
        print(f"Epoch: {epoch:03d}")
        for idx, item in progress(tpipe, header="Train", end=""):
            pass
        if metrics := item.get('metrics', None):
            if lr := metrics.get('lr', None):
                print(f" lr={lr:0.2e}", end="")
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor) and v.numel() > 1:
                    continue
                if k == "lr" or k.startswith("batch"):
                    continue
                if isinstance(v, float):
                    print(f" {k}={v:0.4f}", end="")
                else:
                    print(f" {k}={v}", end="")
        print("")
        
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
        
        if save_best and epoch >= save_best_start:
            if metrics := item.get('metrics', None):
                loss = metrics['loss']
                if loss <= save_best_value:
                    snapshot_state(run_dir, epoch, model, None, metrics)
                    save_best_value = loss
                    last_snapshot = epoch
        
        if snapshot_enabled and epoch >= snapshot_start and last_snapshot != epoch:
            if (epoch-snapshot_start) % snapshot_rate == 0:
                metrics = item.get('metrics', None)
                last_snapshot = epoch
                snapshot_state(run_dir, epoch, model, None, metrics)
        
    if last_snapshot != epoch - 1:
        metrics = item.get('metrics', None)
        snapshot_state(run_dir, epoch-1, model, None, metrics)

    end = datetime.now()
    print(f"finish at: {end.isoformat(sep=' ', timespec='seconds')}")
    
    duration = end - start
    print(f"run time: {duration}")
