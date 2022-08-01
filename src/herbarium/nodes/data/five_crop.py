import torch
import torchvision.transforms.functional as F

from torch.utils.data import IterableDataset
from ..node import Node


class FiveCrop(IterableDataset, Node):
    def __init__(self, inode, height, width):
        IterableDataset.__init__(self)
        Node.__init__(self, inode)
        
        self._height = height
        self._width = width
                
    def __len__(self):
        return len(self.inode) * 5

    def __iter__(self):
        for item in self.inode:
            
            image = item['image']
            
            with torch.no_grad():
                images = F.five_crop(image, (self._height, self._width))
            
            for image in images:
                # duplicate the item dict
                nitem = item.copy()

                nitem['image'] = image
                nitem['orig_width'] = item['image_width']
                nitem['orig_height'] = item['image_height']
                nitem['orig_channels'] = item['image_channels']

                nitem['image_channels'], nitem['image_height'], nitem['image_width'] = image.shape
                
                yield nitem

