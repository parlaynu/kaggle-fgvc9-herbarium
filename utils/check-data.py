#!/usr/bin/env python3
from herbarium.nodes.data import HerbariumDataset


dsroot = "~/Projects/datasets/fgvc9-herbarium-2022"
split = "train"
batch_size = 32

print("creating ds1...")
ds1 = HerbariumDataset(dsroot, split, batch_size, load_images=False)

print("creating ds2...")
ds2 = HerbariumDataset(dsroot, split, batch_size, load_images=False)

print("checking contents...")
for idx, (i1, i2) in enumerate(zip(ds1, ds2)):
    if i1['image_name'] != i2['image_name']:
        print("error: image names don't match")
        print(f"- image1: {i1['image_name']}")
        print(f"- image2: {i2['image_name']}")
        break

print(f"checked {idx} images")

