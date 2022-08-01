from itertools import islice
import math
import torch
from torch.utils.data import IterableDataset

from ..node import Node


class BatchLoader(IterableDataset, Node):
    def __init__(self, inode, batch_size, *, drop_last=False):
        IterableDataset.__init__(self)
        Node.__init__(self, inode)
        
        self._batch_size = batch_size
        self._drop_last = drop_last
        
    def __len__(self):
        return len(self.inode)

    def __iter__(self):
        count = len(self.inode)
        
        print(self._batch_size)

        items = []
        for item in self.inode:
            items.append(item)
            if len(items) == self._batch_size:
                yield items
                items = []
        
        if len(items) > 0 and self._drop_last == False:
            yield items

