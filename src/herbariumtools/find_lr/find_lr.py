
def parse_cmdline():
    import argparse
    import os
    import time
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=16)
    parser.add_argument("-i", "--initial-lr", help="initial learning rate", type=float, default=0.0000001)
    parser.add_argument("-f", "--final-lr", help="final learning rate", type=float, default=0.001)
    parser.add_argument("-n", "--num-steps", help="number of steps", type=int, default=200)
    #parser.add_argument("-l", "--linear", help="take linear steps", action='store_true')
    parser.add_argument('-s', '--state-file', help='state file to load', type=str, default='null')
    parser.add_argument("-v", "--verbose", help="take linear steps", action='store_true')
    parser.add_argument('config_file', help='configuration file to load (- for stdin)', type=str, default=None)
    
    args = parser.parse_args()

    now = datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("snapshots", f"{run_id}-find-lr")

    options = {
        'timestamp': now.isoformat(),
        'run_id': run_id,
        'run_dir': run_dir,
        'batch_size': args.batch_size,
        'initial_lr': args.initial_lr,
        'final_lr': args.final_lr,
        'num_steps': args.num_steps,
        'state_file': args.state_file,
        'verbose': args.verbose,
    }
    
    # default options
    options['num_workers'] = 4
    
    return (args.config_file, options)


def load_and_check(config_file, verbose, **options):
    import sys, os.path
    from pprint import pprint
    import random
    import numpy as np
    import torch
    from herbarium.config import load_config, save_config
    from herbarium.config import instantiate, build_pipeline
    from herbarium.nodes import iter_fwd, get_root
    from herbarium.nodes.train import Trainer
    from herbarium.nodes.logger import Logger
    from herbarium.nodes.scheduler import BatchLinearRamp

    # load and save the config file
    cfg = load_config(config_file, **options)
    
    run_dir = cfg['runtime']['run_dir']
    os.makedirs(run_dir, mode=0o777, exist_ok=False)
    save_config(cfg, run_dir, "find-lr")
    
    # create the root level instances
    instances = {}
    for k, v in cfg.items():
        if k.endswith("_pipeline"):
            continue
        instances[k] = instantiate(cfg[k], instances)
    
    model = instances.get('model')
    optimizer = instances.get('optimizer')
    
    # create the train pipeline
    tpipe = build_pipeline(cfg['train_pipeline'], instances)
    tlogger = None
    
    # change the end of the pipeline
    for n in iter_fwd(tpipe):
        if isinstance(n, Trainer):
            tpipe = n
        if isinstance(n, Logger):
            tlogger = n
            
    if not isinstance(tpipe, Trainer):
        print("Error: no Trainer node in train pipeline")
        sys.exit(1)
    if not isinstance(tlogger, Logger):
        print("Error: no Logger node in train pipeline")
        sys.exit(1)
    
    tpipe = BatchLinearRamp(tpipe, optimizer,
        initial_lr=options['initial_lr'],
        final_lr=options['final_lr'], 
        cycle_len=options['num_steps'],
        track_loss=True
    )
    
    tlogger.inode = tpipe
    tlogger.prefix = "FindLR"
    
    tpipe = tlogger
    
    # print some debug
    print(f"model: {model.fullname}")
    
    print("train pipeline:")
    for n in iter_fwd(tpipe):
        print(f"- {n.fullname}")
    
    print(f"run dir: {cfg['runtime']['run_dir']}")

    print(f"running on {model.device}")
    
    return cfg, tpipe


def run():
    import sys, time
    from pprint import pprint
    from itertools import count, islice
    from herbarium.nodes import iter_fwd
    from herbarium.nodes.logger import Logger
    
    # parse the command line
    config_file, options = parse_cmdline()

    # load the config
    config, tpipe = load_and_check(config_file, **options)
    
    # run the loop
    print()
    print('steps: ', flush=True, end='')
    cur_step = 0
    num_steps = options['num_steps']
    while cur_step < num_steps:
        for item in tpipe:
            print('#', flush=True, end='')
            
            cur_step += 1
            if cur_step >= num_steps:
                break

    print(f': steps={cur_step}')
