import os.path
import torch.nn.functional as F
import torch.nn as nn
import torch

import albumentations as A

from lime import lime_image

from ..node import Node


class Explainer(Node):
    
    def __init__(self, inode, model, transforms):
        super().__init__(inode)

        self._model = model
        self._device = model.device

        if isinstance(transforms, dict):
            transforms = A.Compose(**transforms)
        elif isinstance(transforms, (list, tuple, set)):
            transforms = A.Compose(transforms)
        self._transforms = transforms
        
        self._explainer = lime_image.LimeImageExplainer()
        
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
            image = item['image']

            # get the explanation
            explanation = self._explainer.explain_instance(image,
                                                 self._batch_predict,
                                                 top_labels=5, 
                                                 hide_color=0, 
                                                 batch_size=50,
                                                 num_samples=1000)
            
            item['explanation'] = explanation
            yield item

    def _batch_predict(self, images):
        batch = torch.stack([self._transforms(image=i)['image'] for i in images], dim=0)
        batch = batch.to(self._device, non_blocking=True)
    
        outputs = self._model(batch)
        probs = F.softmax(outputs, dim=1).detach().cpu().numpy()

        return probs
    
