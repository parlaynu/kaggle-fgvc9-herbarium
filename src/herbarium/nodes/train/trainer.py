import torch.nn as nn
import torch
from torch.cuda import amp

from ..node import Node


class Trainer(Node):
    
    def __init__(self, inode, model, criterion, optimizer, *, use_amp=True):
        super().__init__(inode)

        self._model = model
        self._device = model.device
        self._use_amp = False if self._device.type == "cpu" else use_amp
        print(f"train: use_amp: {self._use_amp}")

        self._criterion = criterion
        self._criterion.to(self._device)
        self._optimizer = optimizer
        
    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
    
    def __len__(self):
        return len(self.inode)
    
    def __iter__(self):
        self._model.train()
        
        if self._use_amp:
            scaler = amp.GradScaler()
        
        for items in self.inode:
            items['image'] = inputs = items['image'].to(self._device, non_blocking=True)
            items['target'] = targets = items['target'].to(self._device, non_blocking=True)
            
            self._optimizer.zero_grad()
            
            # NOTE: seeing some strange things after introducing amp... being cautious for
            # now until I have time to test properly
            if self._use_amp:
                with amp.autocast():
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(self._optimizer)
                
                scale = scaler.get_scale()
                scaler.update()
                if scale > scaler.get_scale():
                    # the only reason scale is decreased is if the gradients 
                    # were Inf or NaN... optimizer doesn't get called so 
                    # loop again
                    continue
                
            else:
                outputs = self._model(inputs)
                loss = self._criterion(outputs, targets)
                loss.backward()
                self._optimizer.step()
            
            items['output'] = outputs
            items['metrics'] = {
                'loss' : loss.item(),
                'lr': self._optimizer.param_groups[0]['lr']
            }
            
            yield items

