import sys, os.path
import re
import glob
import math

from PIL import Image
import numpy as np

from torch.utils.data import IterableDataset, get_worker_info

from ..node import Node


class GlobDataset(IterableDataset, Node):

    def __init__(self, dsroot, pattern, batch_size, *, load_images=True):
        IterableDataset.__init__(self)
        Node.__init__(self, None)

        self._dsroot = os.path.expanduser(dsroot)
        self._pattern = pattern
        self._batch_size = batch_size
        self._load_images = load_images
        
        self._image_id_re = re.compile(r'test-(\d+)\.jpg$')

        # load the images
        images = set()
        prefix_len = len(self._dsroot) + 1
        full_pattern = os.path.join(self._dsroot, self._pattern)
        for path in glob.iglob(full_pattern):
            images.add(path[prefix_len:])
        
        self._images = list(images)
        self._images.sort()
        self._length = len(self._images)
        
    def __len__(self):
        return self._length

    def __iter__(self):
        worker_id = 0
        num_workers = 1
        if worker_info := get_worker_info():
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        num_batches = math.ceil(self._length/self._batch_size)
        batches_per_worker = [len([i for i in range(x, num_batches, num_workers)]) for x in range(num_workers)]
        
        start_batch = sum([batches_per_worker[i] for i in range(worker_id)])
        end_batch = start_batch + batches_per_worker[worker_id]
        
        start_idx = start_batch * self._batch_size
        end_idx = min(end_batch * self._batch_size, self._length)
        
        for idx in range(start_idx, end_idx):
            image_name = self._images[idx]
            mo = self._image_id_re.search(image_name)
            if mo is None:
                print(f"no match for {image_name}")
                continue
            
            image_id = int(mo.group(1))
            
            item = { 
                'image_id': image_id,
                'image_name': image_name,
                'image_path' : os.path.join(self._dsroot, image_name),
            }
            
            if self._load_images:
                image_path = item['image_path']
                item['image'] = image = np.asarray(Image.open(image_path).convert("RGB"))
                item['image_height'], item['image_width'], item['image_channels'] = image.shape

            yield item

