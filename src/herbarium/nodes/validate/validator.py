import torch.nn as nn
import torch
from torch.cuda import amp

from ..node import Node


class Validator(Node):
    
    def __init__(self, inode, model, criterion, *, use_amp=True):
        super().__init__(inode)

        self._model = model
        self._device = model.device
        self._use_amp = False if self._device.type == "cpu" else use_amp
        print(f"vdate: use_amp: {self._use_amp}")

        self._criterion = criterion
        self._criterion.to(self._device)
    
    @property
    def device(self):
        return self._device
    
    @property
    def model(self):
        return self._model
    
    def __len__(self):
        return len(self.inode)
        
    def __iter__(self):
        self._model.eval()
        
        for item in self.inode:
            item['image'] = inputs = item['image'].to(self._device, non_blocking=True)
            item['target'] = targets = item['target'].to(self._device, non_blocking=True)

            # NOTE: seeing some strange things after introducing amp... being cautious for
            # now until I have time to test properly
            if self._use_amp:
                with amp.autocast():
                    with torch.no_grad():
                        outputs = self._model(inputs)
                        loss = self._criterion(outputs, targets)
            else:
                with torch.no_grad():
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, targets)
            
            item['output'] = outputs
            item['metrics'] = {"loss": loss.item()}
            
            yield item

