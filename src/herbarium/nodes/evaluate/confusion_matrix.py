from collections import defaultdict

import torch
from ignite.metrics.confusion_matrix import ConfusionMatrix as IConfusionMatrix

from ..node import Node


class ConfusionMatrix(Node):
    
    def __init__(self, inode, num_categories, *, device=None):
        super().__init__(inode)

        if device is None:
            device = torch.device("cpu")
            
        self._cm = IConfusionMatrix(num_categories, average="samples", device=device)
        self.reset()
    
    def reset(self):
        self._cm.reset()
        
    def __len__(self):
        return len(self.inode)
    
    def __iter__(self):
        self.reset()
        
        for idx, item in enumerate(self.inode):
            targets = item['target']
            outputs = item['output']
            
            self._cm.update((output, targets))

            yield item
        
        item['metrics']['confusion_matrix'] = self._cm.compute()
    
