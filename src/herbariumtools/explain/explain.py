#!/usr/bin/env python3

def parse_cmdline():
    import os, argparse
    import time
    from datetime import datetime
    import torch
    from herbarium.config import load_config, save_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--use-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-l', '--sample-limit', help='max batches per epoch (0 = no limit)', type=int, default=0)
    parser.add_argument('config_file', help='configuration file to load (- for stdin)', type=str, default=None)
    parser.add_argument('weights_file', help='weights file to load', type=str, default=None)    
    parser.add_argument('variables', help='key=value variables for template expansion', type=str, nargs='*', default=None)
    
    args = parser.parse_args()
    
    use_gpu = not args.use_cpu
    
    now = datetime.now()
    
    run_id = os.path.basename(args.weights_file)
    run_dir = os.path.dirname(args.weights_file)
    
    options = {
        'timestamp': now.isoformat(sep=' ', timespec='seconds'),
        'run_id': run_id,
        'run_dir': run_dir,
        'num_epochs': -1,
        'time_limit': -1,
        'batch_limit': args.sample_limit,
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
    import os, os.path
    from itertools import count
    from datetime import datetime
    import torch
    import numpy as np
    from PIL import Image
    from skimage.segmentation import mark_boundaries
    
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

    epipe = build_pipeline(cfg['explain_pipeline'], instances)
    
    print("explain pipeline:")
    for n in iter_fwd(epipe):
        print(f"- {n.fullname}")
        
    print(f"running on {model.device}")

    num_right = 0
    num_wrong = 0
    
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/right", exist_ok=True)
    os.makedirs("outputs/wrong", exist_ok=True)
    
    with torch.no_grad():
        for idx, item in enumerate(epipe):
                
            # keys = list(item.keys())
            # keys.sort()
            # for k in keys:
            #     print(f"{k}: {type(item[k])}")
            exp = item['explanation']
            
            print(f"{item['category_id']}: {exp.top_labels}")
            
            save_dir = None
            if item['category_id'] == exp.top_labels[0]:
                num_right += 1
                print(f"right: {num_right}")
                if num_right <= 10:
                    save_dir = "outputs/right"
            else:
                num_wrong += 1
                print(f"wrong: {num_wrong}")
                if num_wrong <= 10:
                    save_dir = "outputs/wrong"
            
            if save_dir is not None:
                img, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
                img_boundary = mark_boundaries(img/255.0, mask)

                path = os.path.join(save_dir, f"{item['image_id']}.jpg")
                pimg = Image.fromarray(img)
                pimg.save(path)
            
                path = os.path.join(save_dir, f"{item['image_id']}-mask.jpg")
                pimg = Image.fromarray((img_boundary*255).astype(np.uint8))
                pimg.save(path)

            if num_right >= 10 and num_wrong >= 10:
                break

    print("")
        
