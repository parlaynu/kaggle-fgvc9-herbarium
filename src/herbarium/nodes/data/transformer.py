import sys
import random
from itertools import cycle

import torch
from torch.utils.data import IterableDataset

import torchvision.transforms as TF
import albumentations as A

from ..node import Node


class TorchvisionTransformer(IterableDataset, Node):
    def __init__(self, inode, transforms):
        IterableDataset.__init__(self)
        Node.__init__(self, inode)
        
        if isinstance(transforms, (list, tuple)):
            transforms = TF.Compose(transforms)

        self._transforms = transforms
        
    def __len__(self):
        return len(self.inode)

    def __iter__(self):
        for item in self.inode:
            # apply the augmentation
            item['image'] = image = self._transforms(item['image'])

            # update the image width/height/channels info in case they've changed
            item['orig_width'] = item['image_width']
            item['orig_height'] = item['image_height']
            item['orig_channels'] = item['image_channels']
            
            if isinstance(image, torch.Tensor):
                item['image_channels'], item['image_height'], item['image_width'] = image.shape
            else:
                item['image_height'], item['image_width'], item['image_channels'] = image.shape
            
            yield item


class AlbumentationsTransformer(IterableDataset, Node):
    def __init__(self, inode, transforms, *, tf_keys=None):
        IterableDataset.__init__(self)
        Node.__init__(self, inode)
        
        if tf_keys is None:
            tf_keys = ['image']
        self._tf_keys = tf_keys
        
        if isinstance(transforms, dict):
            transforms = A.Compose(**transforms)
        elif isinstance(transforms, (list, tuple)):
            transforms = A.Compose(transforms)

        self._transforms = transforms
        
    def __len__(self):
        return len(self.inode)

    def __iter__(self):
        for item in self.inode:
            # apply the augmentation
            tfargs = {k: item[k] for k in item.keys() if k in self._tf_keys}
            item.update(self._transforms(**tfargs))

            # update the image width/height/channels info in case they've changed
            item['orig_width'] = item['image_width']
            item['orig_height'] = item['image_height']
            item['orig_channels'] = item['image_channels']
            
            image = item['image']
            if isinstance(image, torch.Tensor):
                item['image_channels'], item['image_height'], item['image_width'] = image.shape
            else:
                item['image_height'], item['image_width'], item['image_channels'] = image.shape
            
            yield item

