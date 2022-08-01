import warnings
import math
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..node import Node


class OneCycleExponential(Node, _LRScheduler):
    
    def __init__(self, inode, optimizer, *, last_epoch=-1, 
                                            stage0=0, stage1=5, stage2=15,
                                            scale1=10, scale2=0.1,
                                            batch=False):
        
        Node.__init__(self, inode)
        self._batch = batch
        self._stage0 = stage0
        self._stage1 = stage1
        self._stage2 = stage2
        self._scale1 = scale1
        self._scale2 = scale2
        
        steps = self._stage1 - self._stage0
        self._gamma0 = math.pow(self._scale1, 1.0/steps)
        print(f"gamma0: {self._gamma0}")
        
        steps = self._stage2 - self._stage1
        self._gamma1 = math.pow(self._scale2/self._scale1, 1.0/steps)
        print(f"gamma1: {self._gamma1}")
        
        _LRScheduler.__init__(self, optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self._stage2:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        if self.last_epoch >= self._stage1:
            return [group['lr'] * self._gamma1 for group in self.optimizer.param_groups]
            
        if self.last_epoch >= self._stage0:
            return [group['lr'] * self._gamma0 for group in self.optimizer.param_groups]
        
        return [group['lr'] for group in self.optimizer.param_groups]

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

