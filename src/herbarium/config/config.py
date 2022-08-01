import os, time
import copy
from ruamel.yaml import YAML
from jinja2 import Environment, BaseLoader, FileSystemLoader, select_autoescape
import importlib

import random
import numpy as np
import torch


def save_config(config, save_dir, suffix):
    
    filename = f"config-{suffix}.yaml"
    
    cfg_file = os.path.join(save_dir, filename)
    with open(cfg_file, 'w') as f:
        yaml=YAML()
        yaml.dump(config, f)


def load_config(cfg_file, **config_vars):
    # load and render the template
    if cfg_file == "-":
        # load from stdin
        env = Environment(loader=BaseLoader())
        template = env.from_string(sys.stdin.read())
    
    else:
        # load from a file
        cfg_file = os.path.abspath(os.path.expanduser(cfg_file))
        cfg_path = os.path.dirname(cfg_file)
        cfg_name = os.path.basename(cfg_file)
    
        env = Environment(
            loader=FileSystemLoader(cfg_path),
            autoescape=select_autoescape()
        )
        template = env.get_template(cfg_name)
    
    cfg_data = template.render(**config_vars)
    
    # load the yaml
    yaml = YAML(typ='safe')
    config = yaml.load(cfg_data)
    
    # make sure required entries are present
    if config.get('runtime', None) is None:
        config['runtime'] = {}
    runtime = config['runtime']
    if runtime.get('random_seed', None) is None or runtime.get('random_seed') == -1:
        runtime['random_seed'] = int(time.time())
    
    # set the random number seed
    seed = config['runtime']['random_seed']
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    return config


def instantiate(config, instances={}):
    config = copy.deepcopy(config)
    if isinstance(config, (list, tuple)):
        return _instantiate_list(config, instances)
    else:
        return _instantiate_dict(config, instances)


def _instantiate_dict(config, instances):
    # instantiate any nested objects
    for k, v in config.items():
        if isinstance(v, (list, tuple)):
            config[k] = _instantiate_list(v, instances)
        elif isinstance(v, dict):
            config[k] = _instantiate_dict(v, instances)
    
    # if the __instance__ key exists, get the value from the instances
    #   dictionary
    if instance := config.get('__instance__', None):
        return instances[instance]
    
    # if the __target__ key exists, instantiate the object... if not, 
    #   return the dict as is
    if target := config.get('__target__', None):
        return _instantiate_target(target, config)
    
    # nothing else to do... return the config
    return config

    
def _instantiate_list(config, instances):    
    for idx, v in enumerate(config):
        if isinstance(v, (list, tuple)):
            config[idx] = _instantiate_list(v, instances)
        elif isinstance(v, dict):
            config[idx] = _instantiate_dict(v, instances)
            
    return config


def _instantiate_target(target, config):
    del config['__target__']

    tgt_class_path = target.split('.')
    tgt_class_name = tgt_class_path[-1]
    tgt_module_path = '.'.join(tgt_class_path[0:-1])
    
    tgt_module = importlib.import_module(tgt_module_path)
    tgt_class = getattr(tgt_module, tgt_class_name)

    return tgt_class(**config)
