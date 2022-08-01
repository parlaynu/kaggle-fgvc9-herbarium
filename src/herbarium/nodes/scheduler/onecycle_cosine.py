import warnings
import math
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..node import Node


class OneCycleCosine(Node, _LRScheduler):
    
    def __init__(self, inode, optimizer, *, last_epoch=-1, 
                                            peak_epoch=2, final_epoch=10,
                                            peak_scale=10, final_scale=0.1,
                                            batch_mode=False,
                                            warmup_step=0, warmup_scale=0.1):
        
        Node.__init__(self, inode)
        self._peak_scale = peak_scale
        self._final_scale = final_scale / peak_scale
        
        self._peak_epoch = peak_epoch
        self._final_epoch = final_epoch
        
        self._warmup_step = warmup_step
        self._warmup_scale = warmup_scale
        
        self._batch_mode = batch_mode
        if self._batch_mode:
            # if there is a dataloader before this node, then the length of 
            #  the input is the batched length, not the raw length
            self._peak_epoch = int(self._peak_epoch * len(self.inode))
            self._final_epoch = int(self._final_epoch * len(self.inode))
        
        _LRScheduler.__init__(self, optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch <= self._peak_epoch:
            steps = self._peak_epoch
            step = self.last_epoch
            step_scale = 0.5 * (1 - math.cos(step / steps * math.pi))

            self._peak_lrs = [group['initial_lr'] + step_scale * group['initial_lr'] * (self._peak_scale - 1)
                                    for group in self.optimizer.param_groups]
            lrs = self._peak_lrs
        
        elif self.last_epoch <= self._final_epoch:
            steps = self._final_epoch - self._peak_epoch
            step = self.last_epoch - self._peak_epoch
            step_scale = 0.5 * (1 - math.cos(step / steps * math.pi))
            
            self._final_lrs = [peak_lr - step_scale * peak_lr * (1 - self._final_scale)
                                    for peak_lr in self._peak_lrs]
            lrs = self._final_lrs
        
        else:
            lrs = [final_lr for final_lr in self._final_lrs]
        
        if self._warmup_step > 0 and self._warmup_scale != 1.0:
            nlrs = self.last_epoch // self._warmup_step + 1
            for idx, lr in enumerate(lrs):
                if idx < nlrs:
                    continue
                lrs[idx] = lr * self._warmup_scale
        
        return lrs

    def __len__(self):
        return len(self.inode)
    
    def __iter__(self):
        for item in self.inode:
            # if batch scheduling, report batch metrics
            if self._batch_mode:
                item['metrics']['batch_lr'] = item['metrics']['lr']
                item['metrics']['batch_loss'] = item['metrics']['loss']
            
            yield item
            
            # if batch scheduling, take the step here
            if self._batch_mode:
                self.step()
        
        # if epoch scheduling, take the step here
        if self._batch_mode == False:
            self.step()

