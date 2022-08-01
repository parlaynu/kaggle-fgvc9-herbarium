import sys
from collections import defaultdict
import torch

from .meters import AverageMeter

from ..node import Node


class Logger(Node):
    
    def __init__(self, inode, writer, prefix, *, loss_clamp=sys.maxsize):
        super().__init__(inode)
        self.prefix = prefix
        self._writer = writer
        self._loss_clamp = loss_clamp
        self._epoch = -1
        self._global_step = -1
    
    def __len__(self):
        return len(self.inode)
    
    def __iter__(self):
        self._epoch += 1
        
        losses_avg = defaultdict(lambda: AverageMeter())
        lr_avg = AverageMeter()
        
        # log per-batch metrics
        for item in self.inode:
            self._global_step += 1
            
            metrics = item['metrics']
            
            for name, value in metrics.items():
                if name.startswith('batch'):
                    label = ''.join([word.capitalize() for word in name.split('_')])
                    self._writer.add_scalar(f'{self.prefix}/{label}', value, global_step=self._global_step)

            # average the loss and lr across the batch
            for k, v in metrics.items():
                if k.startswith('loss') == False:
                    continue
                losses_avg[k].update(v)
                metrics[k] = losses_avg[k].value()
            
            if lr := metrics.get('lr', None):
                lr_avg.update(lr)
                metrics['lr'] = lr_avg.value()
            
            item['metrics'] = metrics
            
            yield item
        
        # log the metrics once per epoch
        metrics = item['metrics']
        for name, value in metrics.items():
            # don't log multidimensional tensors
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                continue
            
            if name.startswith('batch'): # batch metrics already logged
                continue
            if name.startswith('loss'):  # clamp losses so graphs can be read
                value = min(value, self._loss_clamp)
            
            if isinstance(value, (list, tuple)):
                label = ''.join([word.capitalize() for word in name.split('_')])
                for i, v in enumerate(value):
                    clabel = f'{label}_{i:02d}'
                    self._writer.add_scalar(f'{self.prefix}/{clabel}', v, global_step=self._epoch)
            
            else:
                label = ''.join([word.capitalize() for word in name.split('_')])
                self._writer.add_scalar(f'{self.prefix}/{label}', value, global_step=self._epoch)
            
            self._writer.flush()

