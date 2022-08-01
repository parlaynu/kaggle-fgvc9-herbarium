from itertools import islice
import math
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..node import Node


class BatchLimiter(IterableDataset, Node):
    def __init__(self, inode, batch_limit, batch_size):
        IterableDataset.__init__(self)
        Node.__init__(self, inode)
        
        self._batch_limit = batch_limit
        if self._batch_limit is None:
            self._batch_limit = 0
        self._batch_size = batch_size
        
        if batch_limit is None or batch_limit == 0:
            self._length = len(inode)
        else:
            self._length = min(batch_limit*batch_size, len(inode))
    
    def __len__(self):
        return self._length

    def __iter__(self):
        num_workers = 1
        if worker_info := get_worker_info():
            num_workers = worker_info.num_workers
        
        count = len(self.inode)
        if self._batch_limit > 0:
            count = min(self._batch_limit * self._batch_size, count)
        
        if self._batch_limit > 0 and num_workers > 1:
            batches_per_worker = [len([i for i in range(x, self._batch_limit, num_workers)]) for x in range(num_workers)]
            count = batches_per_worker[worker_info.id] * self._batch_size
            
            if count == 0:
                count = self._batch_size

        for item in islice(self.inode, count):
            yield item
