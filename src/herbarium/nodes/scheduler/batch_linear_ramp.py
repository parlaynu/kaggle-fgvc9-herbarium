import warnings
import math
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..node import Node


class BatchLinearRamp(Node, _LRScheduler):
    
    def __init__(self, inode, optimizer, final_lr, cycle_len, *, 
                        cycle_scale=1.0, initial_lr=None, last_epoch=-1, 
                        start_epoch=0, end_epoch=1_000_000_000, 
                        track_loss=False,
                        **kwargs):
        
        Node.__init__(self, inode)
        
        self._cur_epoch = last_epoch
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._track_loss = track_loss
        
        if initial_lr is None:
            self._initial_lrs = [group['lr'] for group in optimizer.param_groups]
        else:
            self._initial_lrs = [initial_lr for _ in optimizer.param_groups]

        self._final_lr = final_lr
        
        self._cycle_scale = cycle_scale
        self._cur_cycle = 0
        self._cur_step = last_epoch
        self._max_step = cycle_len - 1
        
        _LRScheduler.__init__(self, optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        self._cur_step += 1
        if self._cur_step > self._max_step:
            self._cur_cycle += 1
            self._cur_step = 0

        next_lrs = self._initial_lrs
        
        if self._cycle_scale != 1.0:
            cycle_scale = math.pow(self._cycle_scale, self._cur_cycle)
            next_lrs = [cycle_scale * next_lr for next_lr in next_lrs]

        step_scale = 1.0
        if self._max_step > 0:
            step_scale = self._cur_step / self._max_step
        
        return [min(self._final_lr, lr + step_scale * (self._final_lr - lr)) for lr in next_lrs]

    def __len__(self):
        return len(self.inode)
    
    def __iter__(self):
        self._cur_epoch += 1
        
        for item in self.inode:
            item['metrics']['batch_lr'] = item['metrics']['lr']
            if self._track_loss:
                item['metrics']['batch_loss'] = item['metrics']['loss']
            yield item
            
            if self._cur_epoch >= self._start_epoch and self._cur_epoch < self._end_epoch:
                self.step()

