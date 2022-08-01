import warnings
import math
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..node import Node


class MultiCycleCosine(Node, _LRScheduler):
    
    def __init__(self, inode, optimizer, *, last_epoch=-1, 
                                            stage0=5, stage1=10,
                                            peak_scale=10,
                                            cycle_scale=1.0,
                                            batch=False):
        
        Node.__init__(self, inode)
        
        self._batch = batch
        self._stage0 = stage0
        self._stage1 = stage1
        self._peak_scale = peak_scale

        self._cycle_steps = stage0 + stage1
        self._cycle_scale = cycle_scale
        
        _LRScheduler.__init__(self, optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cycle = self.last_epoch // self._cycle_steps
        cycle_step = self.last_epoch % self._cycle_steps
        
        peak_scale = math.pow(self._cycle_scale, cycle) * self._peak_scale
        
        steps = self._cycle_steps
        
        if cycle_step < self._stage0:
            steps = self._stage0
            scale = (peak_scale - 1.0) * 0.5 * (1.0 - math.cos(cycle_step / steps * math.pi)) + 1.0
        else:
            steps = self._stage1
            cycle_step -= self._stage0
            scale = (peak_scale - 1.0) * 0.5 * (1.0 + math.cos(cycle_step / steps * math.pi)) + 1.0

        return [group['initial_lr'] * scale for group in self.optimizer.param_groups]
        
    def __len__(self):
        return len(self.inode)
    
    def __iter__(self):
        for item in self.inode:
            # if batch scheduling, report batch metrics
            if self._batch:
                item['metrics']['batch_lr'] = item['metrics']['lr']
                item['metrics']['batch_loss'] = item['metrics']['loss']
            
            yield item
            
            # if batch scheduling, take the step here
            if self._batch:
                self.step()
        
        # if epoch scheduling, take the step here
        if self._batch == False:
            self.step()

