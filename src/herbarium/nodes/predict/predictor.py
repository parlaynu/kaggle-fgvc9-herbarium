import os.path
import torch.nn as nn
import torch

from ..node import Node


class Predictor(Node):
    
    def __init__(self, inode, model):
        super().__init__(inode)

        self._model = model
        self._device = model.device
    
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
            
            with torch.no_grad():
                outputs = self._model(inputs)
                preds = torch.argmax(outputs, dim=-1)
            
            # print(f"inputs: {inputs.shape}, outputs: {outputs.shape}, preds: {preds.shape}")
            # print(preds)
            
            item['output'] = outputs
            item['prediction'] = preds
            
            yield item

