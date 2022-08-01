from torch.utils.data import DataLoader as TDataLoader

from ..node import Node


class DataLoader(TDataLoader, Node):

    def __init__(self, inode, **kwargs):
        # hack to work around my poor config parser ;)
        if collate_fns := kwargs.get('collate_fns', None):
            del kwargs['collate_fns']
            kwargs['collate_fn'] = CollateFn(collate_fns)
            
        TDataLoader.__init__(self, inode, **kwargs)
        Node.__init__(self, inode)


class CollateFn:
    def __init__(self, collate_fns):
        self._collate_fns = collate_fns
    
    def __call__(self, batch):
        for cfn in self._collate_fns:
            batch = cfn(batch)
        return batch

