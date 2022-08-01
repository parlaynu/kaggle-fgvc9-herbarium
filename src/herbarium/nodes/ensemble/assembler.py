import sys, os.path
from collections import defaultdict

import torch.nn as nn
import torch

from ..node import Node

reducers = {
    "mean": torch.mean,
    "sum": torch.sum,
    "max": torch.amax,
    "none": None
}

class Assembler(Node):
    
    def __init__(self, inode, *, samples_per_id, reducer="sum"):
        super().__init__(inode)
        self._samples_per_id = samples_per_id
        self._reducer = reducers[reducer]
        
    def __len__(self):
        return int(len(self.inode) / self._samples_per_id)
    
    def __iter__(self):
        cache = defaultdict(list)
        for item in self.inode:
            
            # shortcut for the simple case
            if self._samples_per_id == 1:
                yield item
                continue
            
            # get the image_ids and outputs ... as lists
            image_ids = item['image_id']
            outputs = item['output']

            for idx, (image_id, output) in enumerate(zip(image_ids, outputs)):
                image_id = int(image_id)
                if output.ndim == 1:
                    output = output.unsqueeze(0)
                
                for o in output:
                    cache[image_id].append(o)
                
                if len(cache[image_id]) == self._samples_per_id:
                    # predict based on reduction (or for each in the stack)
                    outputs = torch.vstack(cache[image_id])
                    if self._reducer:
                        outputs = self._reducer(outputs, dim=0)
                    pred = torch.argmax(outputs, dim=-1)
                    del cache[image_id]
                    
                    # generate an 'item' to yield
                    nitem = {}
                    for k, v in item.items():
                        nitem[k] = [v[idx]]
                    nitem['output'] = [outputs]
                    nitem['prediction'] = [pred]
                    
                    yield nitem

