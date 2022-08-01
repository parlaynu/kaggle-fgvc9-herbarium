import sys, os.path
import random, math
import json
from collections import defaultdict
from itertools import islice

from PIL import Image
import numpy as np

from torch.utils.data import IterableDataset, get_worker_info

from ..node import Node


class HerbariumDataset(IterableDataset, Node):

    def __init__(self, dsroot, split, batch_size, *, image_dir="train_images", shuffle=True, shuffle_seed=331, 
                            nfolds=5, vfold=4,
                            load_images=True, excludes=True,
                            as_numpy=True):

        IterableDataset.__init__(self)
        Node.__init__(self, None)
        
        valid_splits = ["train", "val"]
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, not {split}")
        
        self._dsroot = os.path.expanduser(dsroot)
        self._shuffle = shuffle
        self._random = random.Random(shuffle_seed)
        self._batch_size = batch_size
        self._load_images = load_images
        self._as_numpy = as_numpy
        
        excluded = set()
        if excludes:
            exclude_file = os.path.join(self._dsroot, "excludes.txt")
            if os.path.exists(exclude_file):
                with open(exclude_file) as fd:
                    for line in fd:
                        line = line.strip()
                        if len(line) == 0 or line.startswith("#"):
                            continue
                        excluded.add(line)
        
        train_file = os.path.join(self._dsroot, "train_metadata.json")
        with open(train_file) as fd:
            train_data = json.load(fd)
        
        # load the categories
        categories = {}
        for cat in train_data['categories']:
            cat['label'] = f"{cat['genus']} {cat['species']}".lower()
            categories[cat['category_id']] = cat

        # index annotations by category ... and fold them
        annos_by_cat = defaultdict(list)
        for anno in train_data['annotations']:
            annos_by_cat[anno['category_id']].append(anno)
        
        for idx, (cat, annos) in enumerate(annos_by_cat.items()):
            self._random.shuffle(annos)
            folded = [[annos[i] for i in range(x, len(annos), nfolds)] for x in range(nfolds)]
            
            if split == "train":
                del folded[vfold]
                annos_by_cat[cat] = [f for fold in folded for f in fold]
            else:
                annos_by_cat[cat] = folded[vfold]
        
        annotations = {}
        for cat, annos in annos_by_cat.items():
            for anno in annos:
                annotations[anno['image_id']] = anno

        # load the images
        images = []
        for image in train_data['images']:
            image_name = image['file_name']
            if image_name in excluded:
                continue
            
            image_id = image['image_id']
            if anno := annotations.get(image_id, None):
                images.append(image_id)
                anno['image_name'] = image_name
                anno['image_path'] = os.path.join(self._dsroot, image_dir, image_name)
        
        # randomly shuffle the images at least once
        self._random.shuffle(images)
        
        # save the split data
        self._images = images
        self._length = len(self._images)
        self._annotations = annotations
        self._categories = categories
        self._num_categories = max(self._categories.keys()) + 1
        
    def image_ids(self):
        return self._images
    
    def num_categories(self):
        return self._num_categories
    
    def categories(self):
        return self._categories
    
    def category(self, category_id):
        return self._categories[category_id].copy()
    
    def shuffle(self):
        if self._shuffle == False:
            return
        self._random.shuffle(self._images)

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
            image_id = self._images[idx]
            item = self._annotations[image_id].copy()
            
            item['target'] = item['category_id']

            if self._load_images:
                image_path = item['image_path']
                item['image'] = image = Image.open(image_path).convert("RGB")
                if self._as_numpy:
                    item['image'] = np.asarray(image)
                
                item['image_width'], item['image_height'], item['image_channels'] = image.width, image.height, 3

            yield item
