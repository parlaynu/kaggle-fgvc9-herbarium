import copy
from .config import _instantiate_dict


def build_pipeline(config, instances={}):
    config = copy.deepcopy(config)
    
    inode = None
    for idx, v in enumerate(config):
        if inode is not None:
            v['inode'] = inode
        config[idx] = inode = _instantiate_dict(v, instances)
        
    return inode

