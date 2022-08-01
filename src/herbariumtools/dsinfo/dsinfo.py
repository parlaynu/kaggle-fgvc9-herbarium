from collections import defaultdict
from itertools import chain

import numpy as np
from PIL import Image


def run():
    from herbarium.nodes.data import HerbariumDataset
    
    dsroot = "~/Projects/datasets/fgvc9-herbarium-2022"
    
    train_ds = HerbariumDataset(dsroot, "train", 16, image_dir="train_images", shuffle=True, load_images=False, nfolds=3, vfold=0)
    print("train dataset")
    print(f"- num images: {len(train_ds)}")
    print(f"- num categories: {train_ds.num_categories()}")

    vdate_ds = HerbariumDataset(dsroot, "val", 16, image_dir="train_images", shuffle=True, load_images=False, nfolds=3, vfold=0)
    print("vdate dataset")
    print(f"- num images: {len(vdate_ds)}")
    print(f"- num categories: {vdate_ds.num_categories()}")

    # check for non-intersection
    print("checking for intersection")
    train_images = set()
    for item in train_ds:
        train_images.add(item['image_id'])
    vdate_images = set()
    for item in vdate_ds:
        vdate_images.add(item['image_id'])

    vdate_ds1 = HerbariumDataset(dsroot, "val", 16, image_dir="train_images", shuffle=True, load_images=False, nfolds=3, vfold=1)
    vdate_images1 = set()
    for item in vdate_ds1:
        vdate_images1.add(item['image_id'])

    vdate_ds2 = HerbariumDataset(dsroot, "val", 16, image_dir="train_images", shuffle=True, load_images=False, nfolds=3, vfold=2)
    vdate_images2 = set()
    for item in vdate_ds2:
        vdate_images2.add(item['image_id'])
    
    i = vdate_images.intersection(train_images)
    print(f"- intersection: {len(i)}")
    i = vdate_images.intersection(vdate_images1)
    print(f"- intersection: {len(i)}")
    i = vdate_images.intersection(vdate_images2)
    print(f"- intersection: {len(i)}")
    i = vdate_images1.intersection(vdate_images2)
    print(f"- intersection: {len(i)}")
    
    print("counting categories")
    count = defaultdict(lambda: 0)
    
    for item in chain(train_ds, vdate_ds):
        id = item['category_id']
        count[id] += 1

    maxk = 0
    maxv = 0
    mink = 0
    minv = len(vdate_ds)
    
    for k, v in count.items():
        if v > maxv:
            maxk, maxv = k, v
        if v < minv:
            mink, minv = k, v
    
    print(f"max category: {maxk} {maxv}")
    print(f"min category: {mink} {minv}")
    
    # mean0 = mean1 = mean2 = 0
    # std0 = std1 = std2 = 0
    #
    # print("calculating statistics")
    # for idx, item in enumerate(train_ds):
    #     image = item['image']
    #
    #     mean0 = (mean0*idx + np.mean(image[:,:,0]) / 255.0) / (idx+1)
    #     mean1 = (mean1*idx + np.mean(image[:,:,1]) / 255.0) / (idx+1)
    #     mean2 = (mean2*idx + np.mean(image[:,:,2]) / 255.0) / (idx+1)
    #     std0 = (std0*idx + np.std(image[:,:,0]) / 255.0) / (idx+1)
    #     std1 = (std1*idx + np.std(image[:,:,1]) / 255.0) / (idx+1)
    #     std2 = (std2*idx + np.std(image[:,:,2]) / 255.0) / (idx+1)
    #
    #     if idx % 1000 == 0:
    #         print(f"iteration: {idx}")
    #         print(f"    mean: {mean0:0.4f} {mean1:0.4f} {mean2:0.4f}")
    #         print(f"     std: {std0:0.4f} {std1:0.4f} {std2:0.4f}")
    #
    # print(f"final:")
    # print(f"    mean: {mean0:0.4f} {mean1:0.4f} {mean2:0.4f}")
    # print(f"     std: {std0:0.4f} {std1:0.4f} {std2:0.4f}")

    # vdate_ds = HerbariumDataset(dsroot, "val", 16, shuffle=True, load_images=True)
    # print("vdate dataset")
    # print(f"- num images: {len(vdate_ds)}")
    # print(f"- num categories: {vdate_ds.num_categories()}")

